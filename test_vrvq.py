import datetime
import torch
import lightning
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from nets.model import VRVQ
from utils.data import *
from utils.funcs import eval_fid

from args import parse_args


def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('curr_time:', curr_time)

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

    device = args.device
    print("Device:", device)

    data = {
        'cifar10': CIFAR10Data(args),
        'CelebA': CELEBAData(args)
    }[args.dataset]

    test_loader = data.test_dataloader()

    logger = TensorBoardLogger(save_dir=args.save_dir + "vrvq_results/" + str(args.dataset)
                                        + "/base_voca_:" + str(args.num_embeddings)
                                        + "/" + str(args.num_embeddings_min) + "_to_" + str(args.num_embeddings_max),
                               name="base",
                               version=args.seed)
    
    trainer = Trainer(devices=[args.cuda_ind],
                      logger=logger,
                      max_epochs=args.n_epochs,
                      accelerator='gpu')

    model = VRVQ.load_from_checkpoint("saved_model/vrvq/" + str(args.dataset)
                                      + "/base_voca" + str(args.num_embeddings)
                                      + "/seed" + str(args.seed) + '/model.ckpt'
                                      , args=args).to(device)
   
    model.eval()

    trainer.test(model, test_loader)

    fid_result = eval_fid(model, args, model_type="vrvq")
    print(fid_result)


if __name__ == '__main__':
    main()