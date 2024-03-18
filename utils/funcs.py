import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torchvision.utils import save_image

from utils.data import *
torch.set_float32_matmul_precision('high')



def MMD(x, y, device, kernel="multiscale"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.
    ## Implementation based on https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
            torch.zeros(xx.shape).to(device),
            torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)


def dkm(weights, k, args):

    # Parameter settings
    max_iterations = args.max_iter  # Maximum number of iterations
    epsilon = args.epsilon  # Convergence threshold
    temp = args.temp
    device = args.device
    indices = torch.randperm(k)
    num_weights, num_features = weights.size()
    # Distance metric
    MAE = torch.nn.L1Loss()

    # Random initialization
    weights = weights.to(dtype=torch.float32, device=device)
    centroids = weights[indices].to(dtype=torch.float32, device=device)

    for iteration in range(max_iterations):
        # Distance computation
        dist = torch.cdist(weights, centroids, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

        # Attention matrix
        attentions = torch.softmax(-dist/temp, dim=1)
        # Centroid candidate
        aw = torch.mm(attentions.t(), weights)
        a = torch.sum(attentions, dim=0)
        Centroid_tilde = torch.div(aw.t(), a).t()

        # Check convergence
        centroids_diff = MAE(Centroid_tilde, centroids)
        # Update centroids
        centroids = Centroid_tilde.clone()
        closest_indices = torch.argmin(dist, dim=1)
        #print("iter: ", iteration)
        #print("centroids_diff: ", centroids_diff)
        if centroids_diff.item() < epsilon:
            break

    return centroids, closest_indices


def inverse_dkm(model, args):
    weight_src = model.quantize.embed.weight.clone().detach()
    weigh_trg = (torch.randn(args.cluster_target, args.embedding_dim)*0.125).requires_grad_(True)
    min_loss = float('inf')
    best_weigh_trg = None
    if args.cluster_target > 1000:
        kernel = "rbf"
        lr = 5e-4
    else:
        kernel = "multiscale"
        lr = 5e-3
    optimizer = torch.optim.AdamW([weigh_trg], lr=lr)

    for i in range(5000):
        clustered_weight, cluster_weight_idx = dkm(weights=weigh_trg, k=args.num_embeddings, args=args)
        loss = MMD(weight_src, clustered_weight, args.device, kernel=kernel)
        if abs(loss.item()) < abs(min_loss):
            min_loss = loss.item()
            best_weigh_trg = weigh_trg.clone().detach()
        print("epoch:", i, "loss: ", loss.item())

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    if min_loss > 0.01:
        print("retry...")
        weight_src = model.quantize.embed.weight.clone().detach()
        weigh_trg = (torch.randn(args.cluster_target, args.embedding_dim)*0.2).requires_grad_(True)
        optimizer = torch.optim.AdamW([weigh_trg], lr=lr )
        for i in range(5000):
            clustered_weight, cluster_weight_idx = dkm(weights=weigh_trg, k=args.num_embeddings, args=args)
            loss = MMD(weight_src, clustered_weight, args.device, kernel=kernel)
            if abs(loss.item()) < abs(min_loss):
                min_loss = loss.item()
                best_weigh_trg = weigh_trg.clone().detach()
                print("epoch:", i, "loss: ", loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

    print("**optimal loss: ", min_loss)
    return best_weigh_trg


def eval_fid(model, args, model_type="kmvq"):
    model.eval()
    model = model.to(args.device)
    ######### source image (test)#########
    save_dir = args.save_dir + args.img_dir + args.dataset + '/src_test/'
    os.makedirs(save_dir, exist_ok=True)
    # print(f'Saving {len(test_data)} {args.dataset} test images in {save_dir}......')
    data = {
        'cifar10': CIFAR10Data(args),
        'cifar100': CIFAR100Data(args),
        'SVHN': SVHNData(args),
        'MNIST': MNISTData(args),
        'FashionMNIST': FashionMNISTData(args),
        'CelebA': CELEBAData(args)
    }[args.dataset]

    test_loader = data.test_dataloader()
    test_data = data.test_dataset()

    for i, (image, label) in enumerate(test_data):
        file_path = os.path.join(save_dir, f'{args.dataset}_test_{i}.png')
        # Skipped saving as it already exists
        if not os.path.exists(file_path):
            save_image(image, file_path, nrow=1)

    print(f'Saved {len(test_data)} {args.dataset} images in the directory {save_dir}.')

    # print(f'Saving {len(test_data)} {args.dataset} generated images in {save_dir}......')
    if model_type == "kmvq":
        if args.cluster_target == args.num_embeddings:
            save_dir = args.save_dir + args.img_dir + 'kmvq/' + args.dataset + '/voca' + str(args.num_embeddings) + "/seed" + str(args.seed) + '/'
        else:
            save_dir = args.save_dir + args.img_dir + 'kmvq/' + args.dataset + '/voca' + str(args.num_embeddings) + "_to_" + str(args.cluster_target) +  "/seed" + str(args.seed) + '/'
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                x = image.to(args.device)
                x_hat, _, _ = model.clustering(x)
                for j in range(x_hat.size(0)):
                    image = x_hat[j]
                    file_path = os.path.join(save_dir, f'KMVQ_image_{i * args.batch_size_test + j}.png')
                    if not os.path.exists(file_path):
                        save_image(image, file_path, nrow=1)
    elif model_type == "random":
        if args.cluster_target == args.num_embeddings:
            save_dir = args.save_dir + args.img_dir + 'random/' + args.dataset + '/voca' + str(args.num_embeddings) + "/seed" + str(args.seed) + '/'
        else:
            save_dir = args.save_dir + args.img_dir + 'random/' + args.dataset + '/voca' + str(args.num_embeddings) + "_to_" + str(args.cluster_target) +  "/seed" + str(args.seed) + '/'
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                x = image.to(args.device)
                x_hat, _, _ = model.clustering(x)
                for j in range(x_hat.size(0)):
                    image = x_hat[j]
                    file_path = os.path.join(save_dir, f'random_image_{i * args.batch_size_test + j}.png')
                    if not os.path.exists(file_path):
                        save_image(image, file_path, nrow=1)
    elif model_type == "vrvq":
        save_dir = args.save_dir + args.img_dir + 'vrvq/' + args.dataset + '/voca' + str(args.num_embeddings) + "_to_" + str(args.num_embeddings_test) +  "/seed" + str(args.seed) + '/'
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                x = image.to(args.device)
                trg_test = torch.arange(args.num_embeddings_test).unsqueeze(1).to(args.device)
                x_hat_src, _, _, x_hat_trg, _, _ = model(x, trg_test)
                if args.num_embeddings == args.num_embeddings_test: x_hat_trg = x_hat_src
                for j in range(x_hat_trg.size(0)):
                    image = x_hat_trg[j]
                    file_path = os.path.join(save_dir, f'VRVQ_image_{i * args.batch_size_test + j}.png')
                    if not os.path.exists(file_path):
                        save_image(image, file_path, nrow=1)
    print(f'Saved {len(test_data)} generated images in the directory {save_dir}.')

    ## Calculate FID
    save_dir_1 = args.save_dir + args.img_dir + args.dataset + '/src_test/'
    save_dir_2 = save_dir
    result = os.popen("python -m pytorch_fid " + save_dir_1 + ' ' + save_dir_2).read() #python -m pytorch_fid path/to/dataset1 path/to/dataset2
    return result