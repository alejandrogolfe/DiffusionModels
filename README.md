# Latent Diffusion Framework with Scalable Interpolant Transformers (SiT)

This repository provides a generic implementation of a latent diffusion model based on Scalable Interpolant Transformers (SiT).  The framework is designed for class-conditioned image generation tasks and can be adapted to any dataset by modifying the dataloader. The included training example uses a dataset of prostate tissue patches labeled with Gleason grades, but the architecture and pipeline are fully configurable for different domains and classification schemes.

---

## Usage

1. Clone this repository:

```bash
git clone https://github.com/alejandrogolfe/DiffusionModels
```

2. Install dependencies

Make sure you have **Python 3.x** installed. Then, install the required Python packages:


pip install accelerate==1.0.1
pip install openpyxl
pip install wandb
pip install transformers
pip install diffusers
pip install scipy
pip install opencv-python==4.8.0.74

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

3. File Structure

The main components of this project are:

- `sample_ddpm_class_cond.py`: Main script for evaluation.
- `train_ddpm_cond.py`: Main script for training.

4. Configuration
   
The `dataset_params` section in `config/embryo.yaml` specifies the settings for loading and processing embryo images in the BlastDiffusion pipeline. Below is a description of each parameter:

- `load_latents: False`  
  Indicates whether to load precomputed latent representations for the images. `False` means latents will be computed on-the-fly.

- `condition_types: ["context_class"]`  
  Specifies the types of conditions used for conditioning the generative model. Here, `"context_class"` indicates that the model will use the developmental outcome class of each embryo as a condition.

- `z_channels: 4`  
  Number of channels in the latent space representation used by the diffusion model.

- `target: dataset.SiCAPv2.SiCAPv2_loader`  
  The Python class that handles dataset loading.


- `latent_size: 64`  
  Dimensionality of the latent vector for each image.




The `train_params` section in the configuration file defines the key parameters used for training the BlastDiffusion Latent Diffusion Model (LDM). Each parameter is explained below:

- `seed: 1111`  
  The random seed used to ensure reproducibility of training results.

- `task_name: '/workspace/results/output'`  
  Directory where training outputs, checkpoints, and logs will be saved.

- `ldm_batch_size: 24`  
  Batch size used for training the Latent Diffusion Model. A larger batch size may improve training stability but requires more GPU memory.

- `num_samples: 4`  
  Number of samples to generate during validation or evaluation steps.

- `ldm_validate_epochs: 4`  
  Frequency (in epochs) at which the model will be validated using the validation set. In this case, validation occurs every 4 epochs.

- `ldm_lr: 0.0005`  
  Learning rate for training the Latent Diffusion Model.

- `ldm_ckpt_name: 'best_ddpm_ckpt.pth'`  
  Name of the checkpoint file used for saving the best model during training.

- `resume_training: True`  
  Whether to resume training from an existing checkpoint if available. Setting this to `True` allows continuation of interrupted training without starting from scratch.


## Acknowledgment
This repository is mainly based on [StableDiffusion-PyTorch](https://github.com/explainingai-code/StableDiffusion-PyTorch) code base and [SiT](https://github.com/willisma/SiT) . We sincerely thank prior authors on this topic for their code base.

## Citation

Please don't forget to mention us if you use this work


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
