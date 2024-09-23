# RAQ-VAE
This repository contains the official PyTorch/Pytorch-Lightning implementation of **"RAQ-VAE: Rate-Adaptive Vector-Quantized Variational Autoencoder"** [[arXiv](https://arxiv.org/abs/2405.14222)].


--


## Notice

The code provided here corresponds to the version used in the paper submitted in May. Please note that this version is not fully optimized and may still contain bugs or incomplete features. While it is functional and should work as described in the paper, it may require additional debugging or refinement for specific use cases.

We recommend using this code for reference or research purposes, but please proceed with caution if you plan to use it for production-level applications. Contributions and suggestions for improvements are always welcome.

If you have any questions, feel free to contact me at **jeewan0516@kaist.ac.kr**.


## Architectures and Hyperparameters

The model architecture in this code is based on the conventional VQ-VAE framework outlined in the original VQ-VAE paper (van den Oord et al., 2017), with reference to the VQ-VAE-2 implementations available [here](https://github.com/mattiasxu/VQVAE-2), [here](https://github.com/rosinality/vq-vae-2-pytorch), and [here](https://github.com/EugenHotaj/pytorch-generative). We have used ConvResNets from these repositories, which consist of convolutional layers, transpose convolutional layers, and ResBlocks. 

Experiments were conducted on two setups: a server with 4 RTX 4090 GPUs and a machine with 2 RTX 3090 GPUs. The model implementation and training were done using PyTorch (Paszke et al., 2019), PyTorch Lightning (Falcon et al., 2019), and the AdamW optimizer (Loshchilov and Hutter, 2019). Evaluation metrics such as the Structural Similarity Index (SSIM) and Frechet Inception Distance (FID) were computed using the [pytorch-msssim](https://github.com/VainF/pytorch-msssim) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid) libraries, respectively. For further details, please refer to the parameters shown in a table of the paper.

RAQ-VAEs are constructed based on the described VQ-VAE parameters, with additional consideration of each parameter for adaptation.

---

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
  - Options: `vqvae`, `vqvae2`, `vqgan` (to be updated)
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
[1] Seo, Jiwan, and Joonhyuk Kang. "RAQ-VAE: Rate-Adaptive Vector-Quantized Variational Autoencoder." arXiv preprint arXiv:2405.14222 (2024).

# bibtex
@article{seo2024raq,
  title={RAQ-VAE: Rate-Adaptive Vector-Quantized Variational Autoencoder},
  author={Seo, Jiwan and Kang, Joonhyuk},
  journal={arXiv preprint arXiv:2405.14222},
  year={2024}
}
```
