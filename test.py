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
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae/model_based/" + str(args.dataset)
                                                           + "/result"
                                                           + "/voca_size:" + str(args.num_embeddings),
                                                    name='test',
                                                    version=args.seed)
                                                
            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=logger,
                                                    devices=[args.cuda_ind],
                                                    max_epochs=args.n_epochs,
                                                    accelerator='gpu')
            
      
            model = VQVAE_ONE.load_from_checkpoint(args.save_dir + "vqvae/model_based/" + str(args.dataset)
                                                    +"/voca_size:" + str(args.num_embeddings)
                                                    + '/vqvae/version_0'
                                                    + '/checkpoints/epoch=299-step=381300.ckpt',
                                                   args=args).to(device)
            
            model.eval()
            
            if args.cluster_target > args.num_embeddings:
                args.i_clustered_weight = inverse_dkm(model, args)
        
            trainer.test(model, test_loader)
            result = eval_fid(model, args, model_type="mb_vq")
            print(result)
        
        if args.model_type == 'vqvae2':
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae2/model_based/" + str(args.dataset)
                                                        + "/result"
                                                        + "/voca_size:" + str(args.num_embeddings),
                                                    name='test',
                                                    version=args.seed)
                                                
            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=logger,
                                                    devices=[args.cuda_ind],
                                                    max_epochs=args.n_epochs,
                                                    accelerator='gpu')
            
    
            model = VQVAE_ONE.load_from_checkpoint(args.save_dir + "vqvae2/model_based/" + str(args.dataset)
                                                    +"/voca_size:" + str(args.num_embeddings)
                                                    + '/vqvae2/version_0'
                                                    + '/checkpoints/epoch=299-step=381300.ckpt',
                                                args=args).to(device)
            
            model.eval()
            
            if args.cluster_target > args.num_embeddings:
                args.i_clustered_weight = inverse_dkm(model, args)
        
            trainer.test(model, test_loader)
            result = eval_fid(model, args, model_type="mb_vq2")
            print(result)
    
    if args.raq_type == 'dd':
        if args.model_type == 'vqvae':
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae/data_driven/" + str(args.dataset)
                                                        + "/result"
                                                        + "/voca_size:" + str(args.num_embeddings)
                                                        + "/" + str(args.num_embeddings_min) + "_to_" + str(args.num_embeddings_max),
                                                        name='test',
                                                        version=args.seed)
                                                
            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=logger,
                                                    devices=[args.cuda_ind],
                                                    max_epochs=args.n_epochs,
                                                    accelerator='gpu')
            
            model = RAQVAE_ONE.load_from_checkpoint(args.save_dir + "vqvae/data_driven/" + str(args.dataset)
                                                    +"/voca_size:" + str(args.num_embeddings)
                                                    + '/vqvae/version_0'
                                                    + '/checkpoints/epoch=299-step=381300.ckpt',
                                                args=args).to(device)
            
            model.eval()
            
        
            trainer.test(model, test_loader)
            result = eval_fid(model, args, model_type="dd_vq")
            print(result)
        
        if args.model_type == 'vqvae2':
            logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir + "vqvae2/data_driven/" + str(args.dataset)
                                                        + "/result"
                                                        + "/voca_size:" + str(args.num_embeddings)
                                                           + "/" + str(args.num_embeddings_min) + "_to_" + str(args.num_embeddings_max),
                                                    name='test',
                                                    version=args.seed)
                                                
            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=logger,
                                                    devices=[args.cuda_ind],
                                                    max_epochs=args.n_epochs,
                                                    accelerator='gpu')
            
    
            model = VQVAE_ONE.load_from_checkpoint(args.save_dir + "vqvae2/data_driven/" + str(args.dataset)
                                                    +"/voca_size:" + str(args.num_embeddings)
                                                    + '/vqvae2/version_0'
                                                    + '/checkpoints/epoch=299-step=381300.ckpt',
                                                args=args).to(device)
            
            model.eval()
 
            trainer.test(model, test_loader)
            result = eval_fid(model, args, model_type="dd_vq2")
            print(result)
          
          
if __name__ == '__main__':
    main()