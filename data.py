#!/usr/bin/env python
# coding=utf-8
from utils import normalize_np
import staintools
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import lmdb
import pyarrow as pa
from PIL import Image
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from glob import glob
import os
import random
import six
import copy
import pickle

# from utils import Compose, ToTensor
from utils import normalize_np


def read_pl(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x


def random_sample(paths, num=50):
    if len(paths) < num:
        selected_paths = np.random.choice(paths, num, replace=True).tolist()
    else:
        selected_paths = random.sample(paths, num)
    return selected_paths


def row_string_combine(x):
    if str(x['coord_x']).endswith('0'):
        coord_x = str(int(x["coord_x"]))
    else:
        coord_x = str(x['coord_x'])
        print("~~~~~~~~~~~~warning~~~~~~~~~~")
    if str(x['coord_y']).endswith('0'):
        coord_y = str(int(x["coord_y"]))
    else:
        coord_y = str(x['coord_y'])
        print("~~~~~~~~~~~~warning~~~~~~~~~~")
    return coord_x + "x" + coord_y + ".png"


def random_sample_cluster(df, k):
    df = df.groupby("label").apply(
        lambda x: x.sample(k//10) if len(x) >= 5 else x.sample(5, replace=True)
    ).reset_index(drop=True)
    path = df.apply(lambda x: row_string_combine(x), axis=1)
    df["path"] = path
    return df


def get_case_imgs(case_dir):
    return glob(case_dir + "/*.png")


class MILDataset(Dataset):

    def __init__(self, csv_path, lmdb_dir, k, transform=None):
        df = pd.read_csv(csv_path)
        self.transform = transform
        self.case_dirs = list(df.slides_name)
        self.labels = list(df.label)
        self.k = k
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False,
                             readahead=False, meminit=False)
        # with self.env.begin(write=False) as txn:
        #     self.length = pa.deserialize(txn.get(b'__len__'))
            # self.keys = pa.deserialize(txn.get(b'__keys__'))

    def __len__(self):
        # return len(self.case_dirs)
        return len(self.case_dirs)

    def __getitem__(self, index):
        img = None
        samples = []
        env = self.env
        image_paths = get_case_imgs(self.case_dirs[index])
        selected_paths = random_sample(image_paths, num=self.k)
        # selected_paths = copy.deepcopy(random_sample(image_paths))
        # selected_paths = copy.deepcopy(image_paths[-50:])
        with env.begin(write=False) as txn:
            for selected_path in selected_paths:
                flag = selected_path.split("/")[-2] + "_" + selected_path.split("/")[-1]
                byteflow = txn.get(flag.encode())
                imgbuf = pa.deserialize(byteflow)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
                samples.append(img)
        if self.transform is not None:
            trans_samples = []
            for sample in samples:
                sample = self.transform(sample)
                trans_samples.append(sample)
            samples = torch.stack(trans_samples)
        case_name = os.path.split(self.case_dirs[index])[1]
        return samples, self.labels[index], case_name


class VisMILDataset(Dataset):

    def __init__(self, csv_path, lmdb_dir, k, transform=None):
        df = pd.read_csv(csv_path)
        self.transform = transform
        self.case_dirs = list(df.slides_name)
        self.labels = list(df.label)
        self.k = k
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, index):
        img = None
        samples = []
        env = self.env
        image_paths = get_case_imgs(self.case_dirs[index])
        selected_paths = random_sample(image_paths, num=self.k)
        with env.begin(write=False) as txn:
            for selected_path in selected_paths:
                flag = selected_path.split('/')[-2] + "_" + selected_path.split('/')[-1]
                byteflow = txn.get(flag.encode())
                imgbuf = pa.deserialize(byteflow)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
                samples.append(img)

        if self.transform is not None:
            trans_samples = []
            for sample in samples:
                sample = self.transform(sample)
                trans_samples.append(sample)
            samples = torch.stack(trans_samples)
        case_name = os.path.split(self.case_dirs[index])[1]
        return samples, self.labels[index], case_name, selected_paths


class CMILDataset(Dataset):

    def __init__(self, csv_path, cluster_pl_path,
                 lmdb_dir, k, transform=None):
        df = pd.read_csv(csv_path)
        self.cluster_dict = read_pl(cluster_pl_path)
        self.transform = transform
        self.case_dirs = list(df.slides_name)
        self.labels = list(df.label)
        self.k = k
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, index):
        img = None
        samples = []
        env = self.env
        case_name = os.path.split(self.case_dirs[index])[1]
        case_df = self.cluster_dict["dfs"][
            self.cluster_dict["slide_names"].index(case_name)
        ]
        selected_df = random_sample_cluster(case_df, self.k)
        selected_paths = list(selected_df.path)
        with env.begin(write=False) as txn:
            for selected_path in selected_paths:
                flag = case_name + "_" + selected_path
                try:
                    byteflow = txn.get(flag.encode())
                    imgbuf = pa.deserialize(byteflow)
                except TypeError:
                    print(flag)
                    print(case_name)
                    break
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
                samples.append(img)
        if self.transform is not None:
            trans_samples = []
            for sample in samples:
                sample = self.transform(sample)
                trans_samples.append(sample)
            samples = torch.stack(trans_samples)
        return samples, self.labels[index], case_name


if __name__ == "__main__":
    # csv_path = "./total.csv"
    # lmdb_dir = "./lmdb_dir"
    # target_img_path = "../Vahadane/exam_imgs/36616x32520.png"
    csv_path = "./result/ts_bs_res_100_50/tmp.csv"
    lmdb_dir = "/mnt/usb2/share_data/caolei/MIA_xiugao/lmdb_dir_ts_bs"
    # transform = Compose([ToTensor()])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = MILDataset(csv_path, lmdb_dir, k=30, transform=transform)
    # env = dataset.env
    # image_paths = get_case_imgs(dataset.case_dirs[0])
    # selected_paths = random_sample(image_paths, num=dataset.k)
    # selected_paths = copy.deepcopy(random_sample(image_paths))
    # selected_paths = copy.deepcopy(image_paths[-50:])
    # with env.begin(write=False) as txn:
    #     for selected_path in selected_paths:
    #         flag = selected_path.split("/")[-2] + "_" + selected_path.split("/")[-1]
    #         print(flag)
    #         byteflow = txn.get(flag.encode())
    #         imgbuf = pa.deserialize(byteflow)
    #         buf = six.BytesIO()
    #         buf.write(imgbuf)
    #         buf.seek(0)
    #         img = Image.open(buf).convert("RGB")
    # dataset = CMILDataset(csv_path, cluster_pl_path, lmdb_dir,
    #                       k=50, transform=transform)
    # case_name = os.path.split(dataset.case_dirs[0])[1]
    # case_df = dataset.cluster_dict["dfs"][
    #     dataset.cluster_dict["slide_names"].index(case_name)
    # ]
    # selected_df = random_sample_cluster(case_df, dataset.k)
    # selected_paths = list(selected_df.path)
    # env = dataset.env
    # with env.begin(write=False) as txn:
    #     try:
    #         for selected_path in selected_paths:
    #             flag = case_name + "_" + selected_path
    #             print(flag)
    #             byteflow = txn.get(flag.encode())
    #             imgbuf = pa.deserialize(byteflow)
    #             buf = six.BytesIO()
    #             buf.write(imgbuf)
    #             buf.seek(0)
    #             img = Image.open(buf).convert("RGB")
    #     except TypeError:
    #         import ipdb;ipdb.set_trace()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
    # import ipdb;ipdb.set_trace()
    for x in tqdm(dataset):
    #     # x = x.squeeze()
    #     # print(x.shape)
        continue
