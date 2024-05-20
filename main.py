import datetime
import torch
import pytorch_lightning as pl

from nets.model import *
from utils.data import *

from args import parse_args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('curr_time:', curr_time)
    device = args.device
    print("Device:", device)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    pl.seed_everything(args.seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    data = {
        'cifar10': CIFAR10Data(args),
        'CelebA': CELEBAData(args),
        'CelebA_128': CELEBA128Data(args),
        'ImageNet': ImageNetData(args)
    }[args.dataset]

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    
    if args.raq_type == 'mb':
        if args.model_type == 'vqvae':
            # load pre-trained VQ-VAE or train VQ-VAE
            model = VQVAE_ONE(args=args)
            
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae/" + "model_based/" + str(args.dataset)
                                                           + "/voca_size:" + str(args.num_embeddings),
                                                name='vqvae',
                                                version=args.seed) 
            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=logger,
                                                    devices=[args.cuda_ind],
                                                    max_epochs=args.n_epochs,
                                                    callbacks=[
                                                        pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                                    monitor='val_recon_loss',
                                                                                    save_top_k=3,
                                                                                    mode='min')
                                                    ],
                                                    check_val_every_n_epoch=1,
                                                    accelerator='gpu')
        
            trainer.fit(model, train_loader, val_loader)
            args.cluster_target = args.num_embeddings
            best_model = VQVAE_ONE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)
            trainer.test(best_model, test_loader)
            
        elif args.model_type == 'vqvae2':
            model = VQVAE_TWO(args=args)
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae2/" + "model_based/" + str(args.dataset)
                                                           + "/voca_size:" + str(args.num_embeddings),
                                                name='vqvae2',
                                                version=args.seed) 
        
            trainer = pl.Trainer.from_argparse_args(args,
                                logger=logger,
                                devices=[args.cuda_ind],
                                max_epochs=args.n_epochs,
                                callbacks=[
                                pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                monitor='val_recon_loss',
                                                save_top_k=3,
                                                mode='min')
                                ],
                                check_val_every_n_epoch=2,
                                accelerator='gpu')
        
            trainer.fit(model, train_loader, val_loader)
            args.cluster_target = args.num_embeddings
            best_model = VQVAE_TWO.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)
            trainer.test(best_model, test_loader)
            
    elif args.raq_type == 'dd':
        if args.model_type == 'vqvae':
            model = RAQVAE_ONE(args=args)
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae/" + "data_driven/" + str(args.dataset)
                                                        + "/base_voca_:" + str(args.num_embeddings)
                                                        + "/" + str(args.num_embeddings_min) + "_to_" + str(args.num_embeddings_max),
                                                name="vqvae",
                                                version=args.seed)
            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=logger,
                                                    devices=[args.cuda_ind],
                                                    max_epochs=args.n_epochs,
                                                    callbacks=[
                                                        pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                                    monitor='val_recon_loss',
                                                                                    save_top_k=5,
                                                                                    mode='min')
                                                    ],
                                                    check_val_every_n_epoch=2,
                                                    accelerator='gpu')
        
            trainer.fit(model, train_loader, val_loader)
            best_model = RAQVAE_ONE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)
            trainer.test(best_model, test_loader)
            
        elif args.model_type == 'vqvae2':
            model = RAQVAE_TWO(args=args)
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae2/" + "data_driven/" + str(args.dataset)
                                + "/base_voca_:" + str(args.num_embeddings)
                                + "/" + str(args.num_embeddings_min) + "_to_" + str(args.num_embeddings_max),
                            name="vqvae2",
                            version=args.seed)
            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=logger,
                                                    devices=[args.cuda_ind],
                                                    max_epochs=args.n_epochs,
                                                    callbacks=[
                                                    pl.callbacks.ModelCheckpoint(save_weights_only=True,
                                                                    monitor='val_recon_loss',
                                                                    save_top_k=5,
                                                                    mode='min')
                                                    ],
                                                    check_val_every_n_epoch=2,
                                                    accelerator='gpu')
        
            trainer.fit(model, train_loader, val_loader)
            
            best_model = RAQVAE_TWO.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)
            trainer.test(best_model, test_loader)


if __name__ == '__main__':
    main()