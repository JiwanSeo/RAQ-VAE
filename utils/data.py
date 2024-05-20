import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CelebA, ImageNet

import pytorch_lightning as pl


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.batch_size_test = args.batch_size_test

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4, padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor()
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader


    def test_dataset(self):
        transform = T.Compose(
            [
                T.ToTensor()
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=False, transform=transform, download=True)
        return dataset


class CELEBAData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.batch_size_test = args.batch_size_test

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                T.RandomHorizontalFlip()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='train', transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                T.RandomHorizontalFlip()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='valid', transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        transform = T.Compose(
            [
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='test', transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=True
        )
        return dataloader

    def test_dataset(self):
        transform = T.Compose(
            [
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='test', transform=transform, download=True)

        return dataset


class CELEBA128Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.batch_size_test = args.batch_size_test

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.Resize(128),
                T.CenterCrop(128),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                T.RandomHorizontalFlip()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='train', transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.Resize(128),
                T.CenterCrop(128),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                T.RandomHorizontalFlip()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='valid', transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        transform = T.Compose(
            [
                T.Resize(128),
                T.CenterCrop(128),
                T.ToTensor()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='test', transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=True
        )
        return dataloader

    def test_dataset(self):
        transform = T.Compose(
            [
                T.Resize(128),
                T.CenterCrop(128),
                T.ToTensor()
            ]
        )
        dataset = CelebA(root=self.data_dir, split='test', transform=transform, download=True)

        return dataset
    
    
class ImageNetData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.batch_size_test = args.batch_size_test

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        dataset = ImageNet(root=self.data_dir, split='train', transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
                    [
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]
                )
        dataset = ImageNet(root=self.data_dir, split='val', transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        transform = T.Compose(
                    [
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                    ]
                )
        dataset = ImageNet(root=self.data_dir, split='val', transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=True
        )
        return dataloader

    def test_dataset(self):
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor()
            ]
        )
        dataset = ImageNet(root=self.data_dir, split='val', transform=transform)

        return dataset
