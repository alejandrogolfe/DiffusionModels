import os
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
import matplotlib.pyplot as plt
from torch import nn
from accelerate import Accelerator


def sample(accelerator: Accelerator,
           model: nn.Module,
           vae: nn.Module,
           im_size: int,
           scheduler: LinearNoiseScheduler,
           train_config: dict,
           diffusion_model_config: dict,
           diffusion_config: dict,
           dataset_config: dict,
           epoch_idx: int) -> None:
    """
    Realiza el muestreo de imágenes a partir de un modelo de difusión y una entrada de ruido latente.

    Args:
        accelerator (Accelerator): El objeto Accelerator para gestionar el modelo, los tensores y la inferencia distribuida.
        model (nn.Module): El modelo de difusión.
        vae (nn.Module): El modelo VQVAE para decodificar las representaciones latentes a imágenes.
        im_size (int): Tamaño de las imágenes de salida. Este valor debe ser un entero positivo que representa tanto el ancho como la altura de las imágenes generadas.
        scheduler (LinearNoiseScheduler): El programador de ruido que guía el proceso de difusión.
        train_config (dict): Diccionario con los parámetros de configuración del entrenamiento.
        diffusion_model_config (dict): Diccionario con los parámetros de configuración del modelo de difusión (UNet).
        diffusion_config (dict): Diccionario con los parámetros de configuración de la difusión.
        dataset_config (dict): Diccionario con los parámetros de configuración del dataset.
        epoch_idx (int): El índice de la época actual, utilizado para nombrar las imágenes generadas.
        condition_type (str): Tipo de condición ("class" o "text").
        text_tokenizer: Tokenizador de texto, si se usa la condición de texto.
        text_model: Modelo de texto, si se usa la condición de texto.

    Returns:
        None: Esta función no retorna nada. Guarda las imágenes generadas en un archivo.
    """
    model.eval()
    device = accelerator.device
    ########### Muestreo de ruido latente aleatorio ##########
    xt = torch.randn((train_config['num_samples'], dataset_config['z_channels'], im_size, im_size))
    xt = accelerator.prepare(xt)  # Preparar el tensor para `accelerate`
    xt = xt.to(device)
    ###############################################

    ########### Validar la configuración #################

    is_multi_process = accelerator.num_processes > 1

    # Verificar el tipo de condición

    num_samples = train_config['num_samples']
    num_classes = train_config['num_classes']
    sample_classes = torch.randint(0, num_classes, (num_samples,))
    sample_classes = accelerator.prepare(sample_classes)  # Preparar la clase de entrada
    class_labels = diffusion_model_config["params"]["condition_types"]["class_condition_config"]["class_labels"]
    print('Generando imágenes para {}'.format(list(sample_classes.numpy())))
    cond_input =torch.nn.functional.one_hot(sample_classes, num_classes).to(device)

    # Entrada incondicional para la guía libre de clasificador
    uncond_input = cond_input * 0
    uncond_input = accelerator.prepare(uncond_input)



    t = torch.arange(diffusion_config['num_timesteps'], device=device)  # Timesteps en orden ascendente
    t = t.repeat(xt.shape[0], 1)  # Expande a la dimensión (batch_size, num_timesteps)
    t = accelerator.prepare(t)

    ################# Bucle de muestreo ########################
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Obtener la predicción del ruido
        t_i = t[:, i]
        noise_pred_cond = model(xt, t_i, cond_input)
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t_i, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        # Usar el scheduler para obtener x0 y xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, accelerator.prepare(torch.as_tensor(i)))

    with torch.no_grad():
        if is_multi_process:
            ims = vae.module.decode(xt).sample
        else:
            ims = vae.decode(xt).sample

    ims = torch.clamp(ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=2)

    # Convierte el grid a formato numpy para Matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()

    # Crear una figura de Matplotlib para el grid
    fig, axes = plt.subplots(figsize=(12, 12))
    axes.imshow(grid_np)
    axes.axis("off")



    nrow = 2  # Número de imágenes por fila en el grid
    for idx in range(num_samples):
        label = class_labels[sample_classes[idx].item()]  # Obtener el nombre de la clase para la imagen
        row, col = divmod(idx, nrow)  # Calcular posición en una grilla de `nrow`

        # Posiciones para el título encima de la imagen
        x_offset = col * ims.shape[3] + ims.shape[3] // 2
        y_offset = row * ims.shape[2] - 10  # Ajuste para posición del título

        # Añadir el texto del label sobre cada imagen
        plt.text(x_offset, y_offset, label, color="white", fontsize=10, ha="center", va="bottom",
                 backgroundcolor="black")
    # Guardar la imagen con títulos
    output_dir = os.path.join(train_config['task_name'], f'cond_class_samples')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'x{epoch_idx}_grid.png'), bbox_inches='tight')
    plt.close(fig)


def infer_class_or_text_cond(config: dict,
                             accelerator: Accelerator,
                             model: nn.Module,
                             vae: nn.Module,
                             scheduler: LinearNoiseScheduler,
                             epoch_idx: int,
                             ) -> None:
    """
    Realiza la inferencia para la condición de clase o texto dependiendo de la configuración proporcionada.

    Args:
        config (str): Ruta al archivo de configuración en formato YAML.
        accelerator (Accelerator): El objeto Accelerator para gestionar la inferencia distribuida.
        epoch_idx (int): El índice de la época actual.

    Returns:
        None: Esta función no retorna nada. Llama a la función `sample` para generar y guardar las imágenes.
    """

    config_data=config
    diffusion_config = config_data['diffusion_params']
    dataset_config = config_data['dataset_params']
    diffusion_model_config = config_data['unet']
    train_config = config_data['train_params']

    # Crear directorios de salida
    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'], exist_ok=True)

    with torch.no_grad():
        sample(accelerator, model, vae, dataset_config["latent_size"], scheduler, train_config, diffusion_model_config,
               diffusion_config, dataset_config, epoch_idx)




