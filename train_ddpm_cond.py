import argparse
import numpy as np
from tqdm import tqdm
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration
from diffusers import AutoencoderKL
from utils.diffusion_utils import instantiate_from_config
from sample_ddpm_class_cond import infer_class_or_text_cond
from utils.text_utils import *
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from torch.nn import SyncBatchNorm
import yaml


class TrainerLDM:
    """
        Clase para entrenar un modelo Latent Diffusion Model (LDM).
    """

    def __init__(self, config_path: str):
        """
        Inicializa el entrenador cargando la configuración y preparando los componentes principales.

        Args:
            config_path (str): Ruta del archivo de configuración en formato YAML.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            mixed_precision="fp16",
        )

        self._initialize_components()

    def _initialize_components(self):
        """
        Inicializa los componentes del modelo, datasets y optimizador.
        """
        cfg = self.config

        # Inicialización de parámetros

        # Número de núcleos de CPU disponibles

        # Usar el 50% de los núcleos disponibles para num_workers (ajusta según lo que tu sistema pueda manejar)
        num_workers = 0

        self.diffusion_config = cfg['diffusion_params']
        self.train_config = cfg['train_params']
        self.dataset_config = cfg['dataset_params']
        self.diffusion_model_config = cfg['sit']
        self.scheduler = LinearNoiseScheduler(num_timesteps=self.diffusion_config['num_timesteps'],
                                              beta_start=self.diffusion_config['beta_start'],
                                              beta_end=self.diffusion_config['beta_end'])


        # Datasets y dataloaders
        self.im_dataset = instantiate_from_config(self.dataset_config)
        self.data_loader = DataLoader(self.im_dataset, batch_size=self.train_config['ldm_batch_size'], shuffle=True, pin_memory=True, num_workers=num_workers)
        self.data_loader = self.accelerator.prepare(self.data_loader)

        # Modelo
        self.model = instantiate_from_config(self.diffusion_model_config)
        self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
        if os.path.exists(os.path.join(self.train_config['task_name'],
                                       self.train_config['ldm_ckpt_name'])) and self.train_config['resume_training']:
            print('Loaded unet checkpoint')
            state_dict = torch.load(os.path.join(self.train_config['task_name'],
                                                 self.train_config['ldm_ckpt_name']))
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)

        self.model = self.accelerator.prepare(self.model)

        # VAE
        vae = AutoencoderKL.from_pretrained(r"/workspace/data/models/vae")
        self.vae = self.accelerator.prepare(vae)

        # Optimizador y función de pérdida
        self.optimizer = Adam(self.model.parameters(), lr=self.train_config['ldm_lr'])
        self.criterion = torch.nn.MSELoss()

        self.num_epochs = self.train_config['ldm_epochs']
        self.last_fid = float('inf')

    def train(self):
        """
        Ejecuta el proceso de entrenamiento del modelo LDM.
        """
        torch.cuda.empty_cache()
        for epoch in range(self.num_epochs):
            self.model.train()
            torch.cuda.empty_cache()
            self.forward_epoch(epoch)

            # Realiza validación solo en el proceso principal
            if epoch % self.train_config["ldm_validate_epochs"] == 0 and epoch != 0:
                if self.accelerator.is_main_process:  # Solo el proceso principal realiza la validación
                    with torch.no_grad():
                        self.forward_validation(epoch)

                # Realiza inferencia solo en el proceso principal
                if self.accelerator.is_main_process:
                    infer_class_or_text_cond(self.config, self.accelerator, self.model, self.vae, self.scheduler, epoch)

    def forward_epoch(self, epoch: int):
        """
        Ejecuta una época de entrenamiento.

        Args:
            epoch (int): Índice de la época actual.
        """
        losses = []
        for batch in tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
            loss = self.forward_batch(batch)
            losses.append(loss)

        avg_loss = np.mean(losses)
        print(f'Epoch {epoch + 1} completed. Loss: {avg_loss:.4f}')


    def forward_batch(self, batch: dict):
        """
        Procesa un batch de datos, ejecuta el modelo y actualiza los pesos.

        Args:
            batch (dict): Diccionario con los datos del batch.

        Returns:
            float: Valor de la pérdida del batch.
        """
        self.optimizer.zero_grad()
        is_multi_process = self.accelerator.num_processes > 1

        im = batch["data"].float()
        cond = batch["cond"]


        if not self.dataset_config["load_latents"]:
            with torch.no_grad():
                if is_multi_process:
                    im = self.vae.module.encode(im).latent_dist.sample()
                else:
                    im = self.vae.encode(im).latent_dist.sample()


        noise = torch.randn_like(im)
        t = torch.randint(0, self.config['diffusion_params']['num_timesteps'], (im.shape[0],)).to(self.accelerator.device)
        noisy_im = self.scheduler.add_noise(im, noise, t)

        noise_pred = self.model(noisy_im, t, cond)
        loss = self.criterion(noise_pred, noise)
        self.accelerator.backward(loss)
        self.optimizer.step()

        return loss.item()

    def forward_validation(self, epoch: int):
        """
        Ejecuta la validación del modelo usando FID Score y guarda el mejor modelo.

        Args:
            epoch (int): Índice de la época actual.
        """
        best_model_path = os.path.join(self.train_config['task_name'], 'ddpm_'+str(epoch)+'_ckpt.pth')
        torch.save(self.model.state_dict(), best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento de LDM')
    parser.add_argument('--config', dest='config_path', default='/workspace/results/config/sicap.yaml', type=str)
    args = parser.parse_args()
    trainer = TrainerLDM(args.config_path)
    trainer.train()

