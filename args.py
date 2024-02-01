import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='jwseo')

    # dataloader related
    parser.add_argument("--data_dir", type=str, default="../../HDD/dataset/")
    parser.add_argument("--save_dir", type=str, default="../../HDD/vqvae/")
    parser.add_argument("--dataset", type=str, default="CelebA",
                        choices=['cifar10', 'CelebA'])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_test", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # model size
    parser.add_argument("--num_embeddings", type=int, default=128,
                        help="base vocabulary size; number of possible discrete states")
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help="size of the vector of the embedding of each discrete token")
    parser.add_argument("--n_hid", type=int, default=64, help="number of channels controlling the size of the model")

    # Training options
    parser.add_argument('--n_epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=int, default=5e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help='training seed: 10, 42, 170, 682')
    parser.add_argument('--cuda_ind', type=int, default=0,  help='index for cuda device')

    # KMVQ options
    parser.add_argument('--cluster_target', type=int, default=128, help='Codebook clustering taget')
    parser.add_argument('--max_iter', type=int, default=200, help='number of dkm iterations')
    parser.add_argument('--epsilon', type=int, default=1e-8, help='epsilon for softmax function')
    parser.add_argument('--temp', type=int, default=1e-2, help='Softmax temperature of DKM')

    # VRVQ options
    parser.add_argument("--num_embeddings_min", type=int, default=16,
                        help="minimum vocabulary size; number of possible discrete states")
    parser.add_argument("--num_embeddings_max", type=int, default=2048,
                        help="maximum vocabulary size; number of possible discrete states")
    parser.add_argument("--num_embeddings_test", type=int, default=2048,
                        help="Test vocabulary size; number of possible discrete states")

    # directory for FID
    parser.add_argument('--img_dir', type=str, default='imgs/')
    args = parser.parse_args()

    args.device = torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")

    device = args.device
    print("Runnung on CUDA: ", device, "...")

    return args