#!/usr/bin/env python
# coding=utf-8
"""
Transforms for images
"""
import staintools
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import staintools
import pandas as pd

import copy
import math
import sys
import random


# class KLDLoss(nn.Module):

#     def __init__(self):
#         super(KLDLoss, self).__init__()

#     def forward(self, att_val, cluster):
#         kld_loss = 0
#         is_cuda = att_val.device
#         cluster = np.array(cluster)
#         for cls in np.unique(cluster):
#             index = np.where(cluster==cls)[0]

#             kld_loss += torch.nn.functional.kl_div(
#                 att_val[index], torch.ones()
#             )
def reduce_loss(loss, reduction="mean"):
    return loss.mean() if reduction=="mean" else loss.sum() if reduction=="sum" else loss


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, epsilon=0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = nn.functional.log_softmax(output, dim=-1)
        loss = reduce_loss(
            -log_preds.sum(dim=-1), self.reduction
        )
        nll = nn.functional.nll_loss(log_preds, target,
                                     reduction=self.reduction)
        return (1-self.epsilon)*nll+self.epsilon*(loss/c)


def imshow(inp, save_name):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.1, 0.1, 0.1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.savefig(save_name)


def grid_show(tensors, save_name):
    out = make_grid(tensors)
    imshow(out, save_name)


def plot_roc_curve(target, pred, save_path="../external_results/roc.svg"):
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=0.5,
            label="AUC = %0.3f" % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=0.5,
            color="grey", label="Chance", alpha=0.5)
    ax.set(xlim=[-0.03, 1.03], ylim=[-0.03, 1.03])
    ax.legend(loc="best", prop={"size": 15})
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.show()
    plt.savefig(save_path, format="svg")


def plot_pr_curve(target, pred, save_path="../external_results/pr.svg"):
    prec, recall, _ = precision_recall_curve(target, pred)
    pr_score = auc(recall, prec)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, prec, lw=0.5,
            label="PR = %0.3f" % pr_score)
    ax.set(xlim=[-0.03, 1.03], ylim=[-0.03, 1.03])
    ax.legend(loc="best", prop={"size": 15})
    ax.set_xlabel("Recall", fontsize=18)
    ax.set_ylabel("Precision", fontsize=18)
    plt.show()
    plt.savefig(save_path, format="svg")


class ToTensor(object):
    """
    pil list to tensor list
    """
    def __call__(self, pil_list):
        images = [F.to_tensor(x) for x in pil_list]
        return torch.stack(images)


class Normalize(object):
    """Normalize"""
    def __call__(self, pil_list):
        images = [F.normalize(x, (0.5, 0.5, 0.5), (0.1, 0.1, 0.1))
                  for x in pil_list]
        return torch.stack(images)


class RandomHorizontalFlip(object):
    """Horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pil_list):
        images = []
        for x in pil_list:
            if torch.rand(1) < self.p:
                images.append(F.hflip(x))
            else:
                images.append(x)
        return torch.stack(images)


class MyRotationTrans:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


class Cutout(object):
    """Randomly mask out one or more patches from an image"""
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (tensor): Tensor image of size (C, H, W)
        Returns:
            Image with n_holes of dimension lengthxlength cut out of it
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class PixelReg(object):
    """Randomly replace tiles by an image with all pixel values set to
    the mean pixel value of the dataset with a probability of 0.75.
    """
    def __init__(self, mean_pixels, p=0.25):
        self.p = p
        assert mean_pixels is not None
        self.mean_pixels = mean_pixels

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        
        if torch.rand(1) < self.p:
            mask = np.zeros((3, h, w), np.float32)
            mask[0, ...] = self.mean_pixels[0]
            mask[1, ...] = self.mean_pixels[1]
            mask[2, ...] = self.mean_pixels[2]
            _range = np.max(mask) - np.min(mask)
            mask = (mask - np.min(mask)) / _range
            mask = torch.from_numpy(mask)
            return mask
        else:
            return img


def normalize_np(np_array, normalizer):
    source_array = staintools.LuminosityStandardizer.standardize(np_array)
    transformed = normalizer.transform(source_array)
    return transformed


class Compose(object):
    """
    self defined Compose like transforms.Compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pil_list):
        for t in self.transforms:
            pil_list = t(pil_list)
        return pil_list


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """learning rate warmup"""

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def eval_process(epoch, model, criterion, dataloader, device, num_bags=5,
                 aggregate="max"):
    assert aggregate == "mean" or aggregate == "max", "aggregate must be mean or max"
    print("Epoch %d Validation......" % epoch)
    cpu_device = torch.device("cpu")
    model.eval()
    preds_pool = []
    running_loss_pool = []
    for i in range(num_bags):
        print("sampling the %d bag......" % (i+1))
        preds = []
        labels = []
        running_loss = 0.
        for images, targets, _ in tqdm(dataloader):
            images = images.to(device)
            labels.extend(targets.numpy().tolist())
            targets = targets.to(device)
            logits = model(images)
            preds.extend(logits.cpu().numpy()[:, -1].tolist())
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)

        preds_pool.append(preds)
        this_bag_loss = running_loss / len(dataloader.dataset)
        running_loss_pool.append(this_bag_loss)
        print("bag val loss: %.4f" % this_bag_loss)
    
    final_loss_mean = np.mean(running_loss_pool)
    print("Val loss: %.4f" % final_loss_mean)
    preds_pool_array = np.stack(preds_pool)
    if aggregate == "mean":
        final_logits = np.mean(preds_pool_array, axis=0)
    elif aggregate == "max":
        final_logits = np.max(preds_pool_array, aixs=0)
    auc_score = roc_auc_score(labels, final_logits)
    print("Val AUC: %.4f" % auc_score)
    return auc_score, final_loss_mean


@torch.no_grad()
def test_process(model, criterion, dataloader, device, num_bags=5,
                 aggregate="max"):
    assert aggregate == "mean" or aggregate == "max", "aggregate must be mean or max"
    cpu_device = torch.device("cpu")
    model.eval()
    preds_pool = []
    preds_probs_pool = []
    running_loss_pool = []
    all_slide_names = []
    for i in range(num_bags):
        print("sampling the %d bag......" % (i+1))
        preds = []
        preds_probs = []
        labels = []
        running_loss = 0.
        # for ind, (images, targets, slide_names) in enumerate(tqdm(dataloader)):
        #     images = images.to(device)
        #     labels.extend(targets.numpy().tolist())
        #     targets = targets.to(device)
        #     logits = model(images)
        #     pred_probs = torch.softmax(logits, 1)
        #     preds_probs.extend(pred_probs.cpu().numpy()[:, -1].tolist())
        #     preds.extend(logits.cpu().numpy()[:, -1].tolist())
        #     loss = criterion(logits, targets)
        #     running_loss += loss.item() * images.size(0)
        #     if i == 0:
        #         all_slide_names.extend(slide_names)
        try:
            for ind, (images, targets, slide_names) in enumerate(tqdm(dataloader)):
                images = images.to(device)
                labels.extend(targets.numpy().tolist())
                targets = targets.to(device)
                logits = model(images)
                pred_probs = torch.softmax(logits, 1)
                preds_probs.extend(pred_probs.cpu().numpy()[:, -1].tolist())
                preds.extend(logits.cpu().numpy()[:, -1].tolist())
                loss = criterion(logits, targets)
                running_loss += loss.item() * images.size(0)
                if i == 0:
                    all_slide_names.extend(slide_names)
        except RuntimeError:
            print("RuntimeError")
            print(slide_names)
            import ipdb;ipdb.set_trace()
            break
        except TypeError:
            print(slide_names)
            print("TypeError")
            import ipdb;ipdb.set_trace()
            break

        preds_pool.append(preds)
        preds_probs_pool.append(preds_probs)
        this_bag_loss = running_loss / len(dataloader.dataset)
        running_loss_pool.append(this_bag_loss)
        print("bag test loss: %.4f" % this_bag_loss)
    
    final_loss_mean = np.mean(running_loss_pool)
    print("Test loss: %.4f" % final_loss_mean)
    preds_pool_array = np.stack(preds_pool)
    preds_probs_pool_array = np.stack(preds_probs_pool)
    if aggregate == "mean":
        final_logits = np.mean(preds_pool_array, axis=0)
        final_probs = np.mean(preds_probs_pool_array, axis=0)
    elif aggregate == "max":
        final_logits = np.max(preds_pool_array, axis=0)
        final_probs = np.max(preds_probs_pool_array, axis=0)
    auc_score = roc_auc_score(labels, final_logits)
    average_precision = average_precision_score(labels, final_logits)
    print("Test AUC: %.4f" % auc_score)
    print("Test AP: %.4f" % average_precision)
    # plot_roc_curve(labels, final_logits)
    # plot_pr_curve(labels, final_logits)
    df = pd.DataFrame({"slide_name": all_slide_names,
                       "prob": final_probs,
                       "logit": final_logits,
                       "label": labels})

    return auc_score, final_loss_mean, df


def train_process(model, criterion, optimizer, lr_sche, dataloaders,
                  num_epochs, use_tensorboard, device,
                  save_model_path, record_iter, num_bags,
                  aggregate, writer=None):
    model.train()

    best_score = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        lr_scheduler = None
        running_loss = 0.0
        print("====Epoch{0}====".format(epoch))
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(dataloaders["train"]) - 1)
            lr_scheduler = warmup_lr_scheduler(
                optimizer, warmup_iters, warmup_factor
            )

        for i, (images, targets, _) in enumerate(tqdm(dataloaders["train"])):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            running_loss += loss.item() * images.size(0)

            lr = optimizer.param_groups[0]["lr"]

            if (i + 1) % record_iter == 0:
                to_date_cases = (i + 1) * images.size(0)
                tmp_loss = running_loss / to_date_cases
                print("Epoch{0} loss:{1:.4f}".format(epoch, tmp_loss))
                
                if use_tensorboard:
                    writer.add_scalar("Train loss",
                                      tmp_loss,
                                      epoch * len(dataloaders["train"]) + i)
                    writer.add_scalar("lr", lr,
                                      epoch * len(dataloaders["train"]) + i)

        val_auc, val_loss = eval_process(
            epoch, model, criterion, dataloaders["val"],
            device, num_bags, aggregate
        )
        lr_sche.step()

        if val_auc > best_score:
            best_score = val_auc
            best_state_dict = copy.deepcopy(model.state_dict())

        if use_tensorboard:
            writer.add_scalar(
                "validataion AUC", val_auc, global_step=epoch
            )
            writer.add_scalar(
                "validation loss", val_loss, global_step=epoch
            )

        model.train()

    print("Training Done!")
    print("Best Valid AUC: %.4f" % best_score)
    torch.save(best_state_dict, save_model_path)

    print("========Start Testing========")
    model.load_state_dict(best_state_dict)
    test_auc, test_loss, df = test_process(
        model, criterion, dataloaders["test"],
        device, num_bags, aggregate
    )
    if use_tensorboard:
        writer.add_scalar("Test AUC", test_auc, global_step=0)
        writer.close()


def calculate_objective(pred, target):
    target = target.float()
    pred = torch.clamp(pred, min=1e-5, max=1. - 1e-5).squeeze(-1)
    neg_log_likelihood = -1. * (target * torch.log(pred) + (1. - target) * torch.log(1. - pred))
    neg_log_likelihood = neg_log_likelihood.mean()
    return neg_log_likelihood

