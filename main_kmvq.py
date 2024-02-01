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

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    model = KMEANVQ(args=args)

    logger = TensorBoardLogger(save_dir=args.save_dir + "kmeanvq/" + str(args.dataset)
                                        + "/voca_size:" + str(args.num_embeddings),
                               name=str(args.seed))

    checkpoint_callback = ModelCheckpoint(monitor='val_recon_loss',
                                          save_weights_only=True,
                                          save_top_k=10,
                                          mode='min')

    trainer = Trainer(devices=[args.cuda_ind],
                      logger=logger,
                      max_epochs=args.n_epochs,
                      check_val_every_n_epoch=2,
                      callbacks=[checkpoint_callback],
                      accelerator='gpu')

    trainer.fit(model, train_loader, val_loader)
    
    best_model = KMEANVQ.load_from_checkpoint(checkpoint_callback.best_model_path, args=args)
    
    trainer.test(best_model, test_loader)
    

if __name__ == '__main__':
    main()