#!/usr/bin/env python
# coding=utf-8
"""
Model architecture
Author: Lei Cao
"""
import staintools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50

from layergetter import IntermediateLayerGetter


class MaxPool(nn.Module):

    def __init__(self, L=512, D=128, k=50, dropout=False, pretrained=True,
                 output_score=False):
        super(MaxPool, self).__init__()
        self.k = k

        #ResNet34 Backbone
        m1 = resnet34(pretrained=pretrained)
        m_list = []
        for m in m1.children():
            if isinstance(m, nn.AdaptiveAvgPool2d):
                break
            m_list.append(m)
        self.feature_extractor = nn.Sequential(*m_list)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = x.view(-1, self.k, 512)
        x = torch.amax(x, dim=1)
        out = self.classifier(x)
        return out


class AttFPNMIL(nn.Module):

    def __init__(self, k=50, pretrained=True,
                 output_score=False):
        super(AttFPNMIL, self).__init__()
        self.k = k

        #ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2",
                              "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(
                *list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(self.feature_extractor,
                                    self.return_layers)
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attbranch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        scores = self.attbranch(feat4)
        merged_feat = scores[:,0].unsqueeze(1)*feat1 + scores[:,1].unsqueeze(1)*feat2 + scores[:,2].unsqueeze(1)*feat3 + scores[:,3].unsqueeze(1)*feat4
        merged_feat = merged_feat.view(-1, self.k, 512)
        x = torch.amax(merged_feat, dim=1)
        out = self.classifier(x)
        return out


class FPNMIL(nn.Module):

    def __init__(self, k=50, pretrained=True,
                 output_score=False):
        super(FPNMIL, self).__init__()
        self.k = k

        #ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2",
                              "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(
                *list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(self.feature_extractor,
                                    self.return_layers)
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        merged_feat = merged_feat.view(-1, self.k, 512)
        _, top_index = merged_feat.max(dim=1)
        # x = torch.amax(merged_feat, dim=1)
        x, _ = torch.max(merged_feat, dim=1)
        out = self.classifier(x)
        return out, top_index
        # return out


class FPNMIL50(nn.Module):

    def __init__(self, k=50, pretrained=True,
                 output_score=False):
        super(FPNMIL50, self).__init__()
        self.k = k

        #ResNet34 Backbone
        #model = resnet34(pretrained=pretrained)
        model = resnet50(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2",
                              "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(
                *list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(self.feature_extractor,
                                    self.return_layers)
        self.conv64_512 = nn.Conv2d(256, 2048, kernel_size=1)
        self.conv128_512 = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv256_512 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        merged_feat = merged_feat.view(-1, self.k, 2048)
        # x = torch.amax(merged_feat, dim=1)
        x, _ = torch.max(merged_feat, dim=1)
        out = self.classifier(x)
        return out


class FPNMIL_50_sum(nn.Module):

    def __init__(self, k=50, pretrained=True,
                 output_score=False):
        super(FPNMIL_50_sum, self).__init__()
        self.k = k

        #ResNet34 Backbone
        #model = resnet34(pretrained=pretrained)
        model = resnet50(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2",
                              "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(
                *list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(self.feature_extractor,
                                    self.return_layers)
        self.conv64_512 = nn.Conv2d(256, 2048, kernel_size=1)
        self.conv128_512 = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv256_512 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        merged_feat = merged_feat.view(-1, self.k, 2048)
        # x = torch.amax(merged_feat, dim=1)
        x = torch.sum(merged_feat, dim=1)
        out = self.classifier(x)
        return out


class FPNMIL34naive(nn.Module):

    def __init__(self, k=50, pretrained=True,
                 output_score=False):
        super(FPNMIL34naive, self).__init__()
        self.k = k

        #ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2",
                              "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(
                *list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(self.feature_extractor,
                                    self.return_layers)
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512*50, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        x = merged_feat.view(-1, self.k*512)
        out = self.classifier(x)
        return out

class FPNMIL50naive(nn.Module):

    def __init__(self, k=50, pretrained=True,
                 output_score=False):
        super(FPNMIL50naive, self).__init__()
        self.k = k

        #ResNet34 Backbone
        #model = resnet34(pretrained=pretrained)
        model = resnet50(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2",
                              "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(
                *list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(self.feature_extractor,
                                    self.return_layers)
        self.conv64_512 = nn.Conv2d(256, 2048, kernel_size=1)
        self.conv128_512 = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv256_512 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048*50, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        x = merged_feat.view(-1, self.k*2048)
        out = self.classifier(x)
        return out

class FPNMIL_Mean(nn.Module):

    def __init__(self, k=50, pretrained=True,
                 output_score=False):
        super(FPNMIL_Mean, self).__init__()
        self.k = k

        #ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2",
                              "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(
                *list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(self.feature_extractor,
                                    self.return_layers)
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        merged_feat = merged_feat.view(-1, self.k, 512)
        # x = torch.amax(merged_feat, dim=1)
        x = torch.mean(merged_feat, dim=1)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data import MILDataset
    from utils import Compose, ToTensor
    from torchvision import transforms

    # model = MeanPool()
    model = AttFPNMIL()
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5,.5,.5], [.1,.1,.1])
    ])
    csv_path = "./csv_files/tcga/test.csv"
    lmdb_dir = "/home/caolei/code/Lung_Cancer/scripts/lmdb_dir"
    dataset = MILDataset(csv_path, lmdb_dir, 50, transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=6)
    x,y,z = iter(loader).__next__()
    out = model(x)
    import ipdb;ipdb.set_trace()

