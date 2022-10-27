#!/usr/bin/env python
# coding=utf-8
import staintools
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import staintools

import argparse

from data import MILDataset, CMILDataset
from model import MaxPool, FPNMIL, FPNMIL_Mean
from utils import test_process, LabelSmoothingCrossEntropy


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description="Test script")
parser.add_argument(
    "--test_csv_path", type=str, default="./beijing_slide_path.csv",
    help="externel hospital slide path and label csv"
)
parser.add_argument(
    "--lmdb_dir", type=str, default="./lmdb_dir_beijing",
    help="lmdb directory, default ./lmdb_dir_beijing"
)
parser.add_argument(
    "--k", type=int, default=50,
    help="random sample k patches from each slide, default 50"
)
parser.add_argument(
    "--num_workers", type=int, default=16,
    help="num workers, default 16"
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="batch size, default 16"
)
parser.add_argument(
    "--model_path", type=str, default="./results/saved_models/resnet34.pth",
    help="trained model path"
)
parser.add_argument(
    "--output_csv", type=str
)

args = parser.parse_args()


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    ])
    print("========Preparing Dataset========")
    # dataset = MILDataset(args.test_csv_path, args.lmdb_dir, args.k, transform)
    # cluster_pl_path = "/mnt/usb/caolei/cluster_res/cluster.pl"
    # cluster_pl_path = "/mnt/usb/caolei/cluster_res/cluster_beijing.pl"
    # cluster_pl_path = "/mnt/usb/caolei/cluster_res/cluster_jiangsu.pl"
    # dataset = CMILDataset(args.test_csv_path, cluster_pl_path, args.lmdb_dir,
    #                       args.k, transform)
    dataset = MILDataset(args.test_csv_path, args.lmdb_dir, args.k,
                         transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    print("========Dataset Done========")

    print("========Preparing Model========")
    # model = MaxPool(k=args.k, dropout=True, pretrained=False)
    model = FPNMIL(k=args.k)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("========Model Done========")
    model.load_state_dict(torch.load(args.model_path))
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()

    # test_auc, test_loss, df = test_process(
    #     model, criterion, loader, device, num_bags=20, aggregate="mean"
    # )
    test_auc, test_loss, df = test_process(
        model, criterion, loader, device, num_bags=10, aggregate="max"
    )
    # test_auc, test_loss, df = test_process(
    #     model, criterion, loader, device, num_bags=20, aggregate="max"
    # )
    # df.to_csv("../external_results/beijing_pred_results_dropout.csv", index=False)
    # df.to_csv("../external_results/jiangsu_pred_results_dropout_max.csv", index=False)
    # df.to_csv("../external_results/jiangsu_real_max_50_fold5.csv", index=False)
    # df.to_csv("../external_results/beijing_real_max_50_fold5.csv", index=False)
    # df.to_csv("../external_results/cptac_real_max_50_fold1.csv", index=False)
    # df.to_csv("../train_results/internal34/fold5.csv", index=False)
    # df.to_csv("../train_results/internal50/fold1.csv", index=False)
    # df.to_csv("../train_results/internal50/fold2.csv", index=False)
    # df.to_csv("../train_results/internal50/fold3.csv", index=False)
    # df.to_csv("../train_results/internal50/fold4.csv", index=False)
    # df.to_csv("../train_results/internal50/fold5.csv", index=False)
    # df.to_csv("../external_results/jiangsu_res50_agmnet_fold1.csv", index=False)
    # df.to_csv("../external_results/beijing_res34_agmnet_tmp.csv", index=False)
    # df.to_csv("../external_results/beijing_res34_agmnet_tmp2.csv", index=False)
    # df.to_csv("../external_results/beijing_res34_agmnet_tmp3.csv", index=False)
    # df.to_csv("../external_results/beijing_res34_agmnet_tmp4.csv", index=False)
    # df.to_csv("../external_results/beijing_res34_agmnet_tmp5.csv", index=False)
    # df.to_csv("../external_results/cptac_res34_agmnet_fold1.csv", index=False)
    # df.to_csv("./tcga_res/tcga_test_fold1.csv", index=False)
    # df.to_csv("./tcga_res/fold2/tcga_test_fold2.csv", index=False)
    # df.to_csv("./tcga_res/fold3/tcga_test_fold3.csv", index=False)
    # df.to_csv("./tcga_res/fold4/tcga_test_fold4.csv", index=False)
    # df.to_csv("./tcga_res/fold5/tcga_test_fold5.csv", index=False)
    # df.to_csv("./beijing_res/fold1/beijing_fold1.csv", index=False)
    # df.to_csv("./beijing_res/fold2/beijing_fold2.csv", index=False)
    df.to_csv(args.output_csv, index=False)
    # df.to_csv("./beijing_res/beijing_fold3.csv", index=False)
    # df.to_csv("./beijing_res/beijing_fold4.csv", index=False)
    # df.to_csv("./beijing_res/beijing_fold5.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold1.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold2.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold3.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold4.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold5.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_nonorm_fold1.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold2.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold3.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold4.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold5.csv", index=False)
    # df.to_csv("./beijing_res/beijing_fold1_first.csv", index=False)
    # df.to_csv("./beijing_res/beijing_fold2_first.csv", index=False)
    # df.to_csv("./beijing_res/beijing_fold3_first.csv", index=False)
    # df.to_csv("./beijing_res/beijing_fold4_first.csv", index=False)
    # df.to_csv("./beijing_res/beijing_fold5_first.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold1_first.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold2_first.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold3_first.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold4_first.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold5_first.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold1_second.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold2_second.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold3_second.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold4_second_tmp.csv", index=False)
    # df.to_csv("./jiangsu_res/jiangsu_fold5_second.csv", index=False)
    # df.to_csv("./cptac_res/cptac_fold1.csv", index=False)
    # df.to_csv("./cptac_res/cptac_fold2.csv", index=False)
    # df.to_csv("./cptac_res/cptac_fold3.csv", index=False)
    # df.to_csv("./cptac_res/cptac_fold4.csv", index=False)
    # df.to_csv("./cptac_res/cptac_fold5.csv", index=False)


if __name__ == "__main__":
    main(args)

