import sys
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import models, transforms, datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, df, transform=None):
        paths, lefts1, tops1, rights1, bottoms1, lefts2, tops2, rights2, bottoms2 = df["path"], df["left1"], df["top1"], df["right1"], df["bottom1"], df["left2"], df["top2"], df["right2"], df["bottom2"]
        self.transform = transform
        self.paths = paths.values
        self.lefts1 = lefts1.values
        self.tops1 = tops1.values
        self.rights1 = rights1.values
        self.bottoms1 = bottoms1.values
        self.lefts2 = lefts2.values
        self.tops2 = tops2.values
        self.rights2 = rights2.values
        self.bottoms2 = bottoms2.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x = Image.open(path)
        left1, top1, right1, bottom1 = self.lefts1[idx], self.tops1[idx], self.rights1[idx], self.bottoms1[idx]
        left2, top2, right2, bottom2 = self.lefts2[idx], self.tops2[idx], self.rights2[idx], self.bottoms2[idx]

        if self.transform:
            x = self.transform(x)  # xはtorch型

        x = x.repeat(3, 1, 1)  # grayからRGBの形に変換。(3,300,300)

        boxes = [[left1, top1, right1, bottom1]]
        if str(left2)!="nan":
            #2個目の病変があるケース
            boxes.append([left2, top2, right2, bottom2])

        return x, torch.tensor(boxes, dtype=torch.float)

def preprocess_df(df, originalSize, size, Dir):
    df["path"] = Dir + df["file"]
    # modify the bbox as we resized the input image from originalSize to the size
    df["left1"] = df["left1"] * size / originalSize
    df["top1"] = df["top1"] * size / originalSize
    df["right1"] = df["right1"] * size / originalSize
    df["bottom1"] = df["bottom1"] * size / originalSize
    df["left2"] = df["left2"] * size / originalSize
    df["top2"] = df["top2"] * size / originalSize
    df["right2"] = df["right2"] * size / originalSize
    df["bottom2"] = df["bottom2"] * size / originalSize
    return df

def collate_fn(batch):
    return tuple(zip(*batch))

#バグ修正版。
def visualize(model, local_rank, dataloader, dataframe, numSamples, saveDir, numDisplay=2):

    torch.cuda.empty_cache()
    model.eval()

    #少なくともnumSamples以上のデータ数は出力する仕様にする
    num = 0

    for images, targets in dataloader:
        images = list(img.to(local_rank) for img in images)  #list(img.to(device) for img in images)
        outputs = model(images)

        for ind, image in enumerate(images):
            #image shape is (3,300,300)
            image = image.permute(1, 2, 0).cpu().numpy() #bboxを白黒表示したいなら、この代わりに、image = image[0].cpu().numpy() #(300,300) でよい
            image = Image.fromarray((image * 255).astype(np.uint8))
            trueBoxes = targets[ind]

            #outputs
            boxes = outputs[ind]["boxes"].data.cpu().numpy()
            scores = outputs[ind]["scores"].data.cpu().numpy()
            #labels = outputs[ind]["labels"].data.cpu().numpy()

            # filtering by the top numDisplay
            boxes = boxes[:numDisplay].astype(np.int32)
            scores = scores[:numDisplay]

            #visualization
            for i, box in enumerate(boxes):
                draw = ImageDraw.Draw(image)
                score = scores[i]
                label = "score:{:.2f}".format(score)
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

                #fnt = ImageFont.truetype('/content/mplus-1c-black.ttf', 20)
                #fnt = ImageFont.truetype("arial.ttf", 10) #size
                fnt = ImageFont.load_default()
                text_w, text_h = fnt.getsize(label)
                draw.rectangle([box[0], box[1], box[0]+text_w, box[1]+text_h], fill="red")
                draw.text((box[0], box[1]), label, font=fnt, fill='white')

            #visualization
            for i, box in enumerate(trueBoxes):
                draw = ImageDraw.Draw(image)
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="blue", width=3)

            # set the savePath
            fileName = dataframe["file"][num+ind]  #fileName = dataframe["file"][ind]
            savePath = saveDir + fileName

            # 画像を保存したい時用
            #image.save(f"resample_test{str(i)}.png")
            image.save(savePath)

        num += len(images)
        if num >= numSamples:
            break


def train(dataloader, model, local_rank, optimizer):
    total = 0
    sum_loss = 0

    #training
    model.train()
    for images, bboxes in dataloader:
        images = list(img.to(local_rank) for img in images)  #list(image.to(device) for image in images)
        batch = len(images)

        #targets = [{"boxes": bbox.to(device), "labels": torch.ones(len(bbox), dtype=torch.int64).to(device)} for bbox in bboxes]
        targets = [{"boxes": bbox.to(local_rank), "labels": torch.ones(len(bbox), dtype=torch.int64).to(local_rank)} for bbox in bboxes]

        # return from the model
        loss_dict = model(images, targets)  # 返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）
        losses = sum(loss for loss in loss_dict.values())
        loss = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total += batch
        sum_loss += loss

    return sum_loss / total


def valid(dataloader, model, local_rank):
    total = 0
    sum_loss = 0

    # validation
    #model.train() あえてつける必要はないが、結局これと同じ。with torch.no_grad(): をつけている間は誤差逆伝搬はなされないのでOK。
    model.train()
    for images, bboxes in dataloader:
        images = list(img.to(local_rank) for img in images)  #list(image.to(device) for image in images)
        batch = len(images)

        #targets = [{"boxes": bbox.to(device), "labels": torch.ones(len(bbox), dtype=torch.int64).to(device)} for bbox in bboxes]
        targets = [{"boxes": bbox.to(local_rank), "labels": torch.ones(len(bbox), dtype=torch.int64).to(local_rank)} for bbox in bboxes]

        # return from the model
        loss_dict = model(images, targets)  # 返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）
        losses = sum(loss for loss in loss_dict.values())
        loss = losses.item()

        total += batch
        sum_loss += loss

    return sum_loss / total


def mIoU(dataloader, model, local_rank, numDisplay=2):

    total = 0
    sum_miou = 0

    model.eval()
    for images, targets in dataloader:
        images = list(img.to(local_rank) for img in images)  #list(image.to(device) for image in images)
        batch = len(images)
        outputs = model(images)

        #calculate iou with targets and outputs
        for i in range(batch):
            trueBoxes = targets[i]
            boxes = outputs[i]["boxes"].data.cpu()  #.numpy()
            scores = outputs[i]["scores"].data.cpu()  #.numpy()

            # filtering by the top numDisplay
            boxes = boxes[:numDisplay]
            scores = scores[:numDisplay]

            #deal with when no bbox is predicted. When len(boxes)==0, sum_miou += 0
            if len(boxes)>0:
                # calculate iou with targets and outputs
                bboxIou = box_iou(boxes, trueBoxes)
                maxIou = bboxIou.max(axis=0).values.numpy()  #calculate the maximum IoU for each trueBox
                #maxIou = np.array(maxIou.values)
                meanIou = maxIou.mean()  #sum(maxIou)/len(maxIou)
                sum_miou += meanIou

        total += batch

    return sum_miou / total


def mAP(dataloader, model, local_rank, numDisplay=2):

    total = 0
    sum_ap = 0

    iou_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    model.eval()
    for images, targets in dataloader:
        images = list(img.to(local_rank) for img in images)  #list(image.to(device) for image in images)
        batch = len(images)
        outputs = model(images)

        #calculate iou with targets and outputs
        for ind in range(batch):
            trueBoxes = targets[ind]
            boxes = outputs[ind]["boxes"].data.cpu()  #.numpy()
            scores = outputs[ind]["scores"].data.cpu()  #.numpy()

            # filtering by the top numDisplay
            boxes = boxes[:numDisplay]
            scores = scores[:numDisplay]

            n_tps = np.zeros(len(iou_thresholds))

            # calculate iou with targets and outputs
            bboxIou = box_iou(boxes, trueBoxes)
            maxIou = bboxIou.max(axis=1).values.numpy()  #calculate the maximum IoU for each bbox, not trueBox
            gt_indices = bboxIou.argmax(axis=1)
            gt_used = np.zeros((len(iou_thresholds), len(trueBoxes)), dtype=bool)

            ap = 0
            for k, (iou, conf) in enumerate(zip(maxIou, scores)):
                for i in range(len(iou_thresholds)):
                    if not gt_used[i, gt_indices[k]] and iou >= iou_thresholds[i]:
                        n_tps[i] += 1
                        gt_used[i, gt_indices[k]] = True
                # TP + FP = (# of predictions), TP + FN = len(gt_bbox)
                # thus TP + FP + FN = (# of predictions) + len(gt_bbox) - TP
                prec = n_tps / ((k + 1) + len(trueBoxes) - n_tps)
                ap = prec.mean()

            sum_ap += ap

        total += batch

    return sum_ap / total
