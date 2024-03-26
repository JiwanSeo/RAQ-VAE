import datetime
import torch
import lightning

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

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
    if args.cuda_ind == 0:
        args.seed = 10
    elif args.cuda_ind == 1:
        args.seed = 42
    elif args.cuda_ind == 2:
        args.seed = 170
    elif args.cuda_ind == 3:
        args.seed = 682
    else:
        raise
    lightning.seed_everything(args.seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    data = {
        'cifar10': CIFAR10Data(args),
        'CelebA': CELEBAData(args)
    }[args.dataset]

    test_loader = data.test_dataloader()

    vq_model = KMEANVQ.load_from_checkpoint("../saved_model/kmvq/" + str(args.dataset)
                                            +  "/voca" + str(args.num_embeddings)
                                            + "/seed" + str(args.seed) + '/model.ckpt'
                                            , args=args).to(device)
    
    #  "/checkpoints/epoch=77-step=99138.ckpt"
    #  "/checkpoints/epoch=33-step=43214.ckpt"
    #  "/checkpoints/epoch=413-step=526194.ckpt"
    #  "/checkpoints/epoch=353-step=449934.ckpt"
    model = PixelSNAIL_KMVQ.load_from_checkpoint(args.save_dir  + "pixelsnail/" + str(args.dataset)
                                                 + "/base_voca_:" + str(args.num_embeddings)
                                                 + "/" + "to_" + str(args.cluster_target)
                                                 + "/kmvq_v2/" + "version_" + str(args.seed) 
                                                 + "/checkpoints/epoch=353-step=449934.ckpt",
                                                 n_codes=args.cluster_target,
                                                 n_filters=128,
                                                 n_res_blocks=10,
                                                 n_snail_blocks=3,
                                                 n_condition_blocks=5,
                                                 vq_model=vq_model,
                                                 args=args)
    
    logger = TensorBoardLogger(save_dir=args.save_dir + "pixelsnail_results/kmvq/" + str(args.dataset) + "/seed_" + str(args.seed) + "/voca_size:" + str(args.num_embeddings),
                               name="target_" + str(args.cluster_target))
    
    trainer = Trainer(devices=[args.cuda_ind],
                      logger=logger,
                      max_epochs=args.n_epochs,
                      accelerator='gpu')

    trainer.test(model, test_loader)
    

if __name__ == '__main__':
    main()