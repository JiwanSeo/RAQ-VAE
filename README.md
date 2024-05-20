# RAQ-VAE
This repository contains the official PyTorch/Pytorch-Lightning implementation of **"RAQ-VAE: Rate-Adaptive Vector-Quantized Variational Autoencoder"** (**).

> **Abstract:**   Vector Quantized Variational AutoEncoder (VQ-VAE) is an established technique in machine learning for learning discrete representations across various modalities. However, its scalability and applicability are limited by the need to retrain the model to adjust the codebook for different data or model scales. We introduce the Rate-Adaptive VQ-VAE (RAQ-VAE) framework, which addresses this challenge with two novel codebook representation methods: a model-based approach using a clustering-based technique on an existing well-trained VQ-VAE model, and a data-driven approach utilizing a sequence-to-sequence (Seq2Seq) model for variable-rate codebook generation. Our experiments demonstrate that RAQ-VAE achieves effective reconstruction performance across multiple rates, often outperforming conventional fixed-rate VQ-VAE models. This work enhances the adaptability and performance of VQ-VAEs, with broad applications in data reconstruction, generation, and computer vision tasks.



## Requirements

Before running this project, make sure to set up your environment by following these pre-installation steps. This will ensure that all necessary dependencies are installed correctly.

Use the requirements.txt file to install all the necessary dependencies:

    pip install -r requirements.txt


## Trainig

Once you have set up your environment, you can run the training script. Below is an example of how to execute the script with specific arguments.

#### Exapmple command: Model-based RAQ-VAE / baseline model: VQ-VAE-2

    python main.py --dataset CelebA --raq_type mb --model_type vqvae2 --n_epochs 100 --seed 10 --cuda_ind 

- Make sure your dataset paths specified in args.py are correct.
- Adjust the batch size and other hyperparameters in args.py as needed.
- The script will save model checkpoints and logs to the specified save_dir.


## Evaluation

After training the model, you can evaluate it using the test.py script. Below is an example command to run the evaluation:

#### Exapmple command: Model-based RAQ-VAE / baseline model: VQ-VAE-2

    python test.py --dataset CelebA --raq_type mb --model_type vqvae2 --seed 10 --cuda_ind 0

The evaluation script relies on the checkpoint saved during training. Make sure the path specified in the script matches the location of your checkpoint.



## Command Line Arguments Guide


### Dataloader Related Arguments

- `--data_dir` (str): Path to the dataset directory.
  - Default: `../../HDD/dataset/`
- `--save_dir` (str): Directory to save model checkpoints and logs.
  - Default: `../../HDD2/raqvae/`
- `--dataset` (str): Dataset to use for training and evaluation.
  - Options: `cifar10`, `CelebA`, `CelebA_128`, `ImageNet`
  - Default: `CelebA`
- `--batch_size` (int): Batch size for training.
  - Default: `128`
- `--batch_size_test` (int): Batch size for testing.
  - Default: `64`
- `--num_workers` (int): Number of workers for the dataloader.
  - Default: `8`

### Model Size Related Arguments

- `--raq_type` (str): RAQ type: 'mb' for model-based or 'dd' for data-driven.
  - Options: `mb`, `dd`
  - Default: `mb`
- `--model_type` (str): Model type to use.
  - Options: `vqvae`, `vqvae2`, `vqgan`
  - Default: `vqvae`
- `--num_embeddings` (int): Number of embeddings (vocabulary size).
  - Default: `256`
- `--embedding_dim` (int): Dimension of each embedding vector.
  - Default: `64`
- `--n_hid` (int): Number of hidden channels controlling the model size.
  - Default: `64`

### Training Options

- `--n_epochs` (int): Number of training epochs.
  - Default: `300`
- `--lr` (float): Learning rate.
  - Default: `5e-4`
- `--seed` (int): Random seed for reproducibility.
  - Default: `0`
- `--cuda_ind` (int): Index of the CUDA device to use.
  - Default: `0`

### Model-based RAQ-VAE Options

- `--cluster_target` (int): Target number of clusters for codebook clustering.
  - Default: `512`
- `--max_iter` (int): Maximum number of DKM iterations.
  - Default: `200`
- `--epsilon` (float): Epsilon for the softmax function.
  - Default: `1e-8`
- `--temp` (float): Softmax temperature for DKM.
  - Default: `1e-2`

### Data-driven RAQ-VAE Options

- `--num_embeddings_min` (int): Minimum vocabulary size.
  - Default: `32`
- `--num_embeddings_max` (int): Maximum vocabulary size.
  - Default: `2048`
- `--num_embeddings_test` (int): Vocabulary size for testing.
  - Default: `512`

### Directory for FID evaluation

- `--img_dir` (str): Directory to save images for FID calculation.
  - Default: `imgs/`




---



```
# Citation
[1] 

# bibtex
```
