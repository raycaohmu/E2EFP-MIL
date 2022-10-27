#!/usr/bin/env python
# coding=utf-8
import staintools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
import os
import warnings
import pickle
warnings.filterwarnings("ignore")

from data import MILDataset
from utils import train_process, MyRotationTrans, grid_show, Cutout, PixelReg
from model import MaxPool, AttFPNMIL, FPNMIL, FPNMIL_Mean, FPNMIL50, FPNMIL50naive, FPNMIL34naive, FPNMIL_50_sum


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser(description="Running script")

parser.add_argument(
    "--train_csv_path", type=str, default="./train.csv",
    help="train case path csv, default ./train.csv"
)
parser.add_argument(
    "--val_csv_path", type=str, default="./val.csv",
    help="val case path csv, default ./val.csv"
)
parser.add_argument(
    "--test_csv_path", type=str, default="./test.csv",
    help="test case path csv, default ./test.csv"
)
parser.add_argument(
    "--lmdb_dir", type=str, default="/home/caolei/code/Lung_Cancer/scripts/lmdb_dir"
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="batch size, default 16"
)
parser.add_argument(
    "--num_workers", type=int, default=16,
    help="num workers, default 16"
)
parser.add_argument(
    "--k", type=int, default=50,
    help="random sample k patches from each slide, default 50"
)
parser.add_argument(
    "--lr", type=float, default=0.001,
    help="learning rate, default 0.001"
)
parser.add_argument(
    "--momentum", type=float, default=0.9,
    help="SGD momentum, default 0.9"
)
parser.add_argument(
    "--weight_decay", type=float, default=5e-4,
    help="SGD weight decay, default 5e-4"
)
parser.add_argument(
    "--gamma", type=float, default=0.1,
    help="StepLR gamma value, default 0.1"
)
parser.add_argument(
    "--num_epochs", type=int, default=500,
    help="num of epochs, default 500"
)
parser.add_argument(
    "--save_model_path", type=str,
    default="./results/saved_models/resnet34.pth",
    help="model saving path"
)
parser.add_argument(
    "--record_iter", type=int, default=10,
    help="print record iter, default 10"
)
parser.add_argument(
    "--num_bags", type=int, default=5,
    help="num of bags sampled from a WSI during val or testing, default 5"
)
parser.add_argument(
    "--aggregate", type=str, default="mean",
    help="aggregate method during val or testing, default mean"
)
parser.add_argument(
    "--use_tensorboard", action="store_true",
    help="whether use use tensorboard"
)
parser.add_argument(
    "--logdir", type=str, default="./logs",
    help="tensorboard log dir, default ./logs"
)

args = parser.parse_args()
os.makedirs("./results/saved_models", exist_ok=True)


def main(args):
    # transform = Compose([ToTensor()])
    with open("./mean_pixel.pkl", "rb") as f:
        mean_pixels = pickle.load(f)
    rotation = MyRotationTrans([0, 90, 180, 270])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        rotation,
        transforms.ColorJitter(brightness=0.075, saturation=0.075, hue=0.075),
        transforms.ToTensor(),
        Cutout(1, 100),
        PixelReg(mean_pixels=mean_pixels),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    ])
    print("========Preparing Dataset========")
    train_dataset = MILDataset(args.train_csv_path, args.lmdb_dir,
                               args.k, transform)
    # cluster_pl_path = "/mnt/usb/caolei/cluster_res/cluster.pl"
    # train_dataset = CMILDataset(args.train_csv_path, cluster_pl_path,
    #                             args.lmdb_dir, args.k, transform)
    # grid_show(train_dataset[0][0], "tmp5.png")
    # import ipdb;ipdb.set_trace()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
    # val_dataset = CMILDataset(args.val_csv_path, cluster_pl_path,
    #                           args.lmdb_dir, args.k, transform=val_transform)
    val_dataset = MILDataset(args.val_csv_path, args.lmdb_dir,
                             args.k, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    # test_dataset = CMILDataset(args.test_csv_path, cluster_pl_path,
    #                            args.lmdb_dir, args.k, transform=val_transform)
    test_dataset = MILDataset(args.test_csv_path, args.lmdb_dir,
                              args.k, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    print("========Dataset Done========")

    print("========Preparing Model========")
    #model = MaxPool(k=args.k, dropout=True)
    # model = AttFPNMIL(k=args.k)
    # model = FPNMIL(k=args.k)
    # model = FPNMIL_Mean(k=args.k)
    # model = FPNMIL50(k=args.k)
    #model = FPNMIL50naive(k=args.k)
    # model = FPNMIL34naive(k=args.k)
    model = FPNMIL_50_sum(k=args.k)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("========Model Done========")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[100, 300], gamma=args.gamma
    # )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200], gamma=args.gamma
    )
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    
    print("========Start Training========")
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    train_process(model=model, criterion=criterion, optimizer=optimizer,
                  lr_sche=lr_scheduler, dataloaders=dataloaders, writer=writer,
                  num_epochs=args.num_epochs, use_tensorboard=args.use_tensorboard,
                  device=device, save_model_path=args.save_model_path,
                  record_iter=args.record_iter, num_bags=args.num_bags,
                  aggregate=args.aggregate)


if __name__ == "__main__":
    main(args)

