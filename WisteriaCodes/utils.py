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
from scipy import integrate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, df, transform=None, inDim=3):
        paths, lefts1, tops1, rights1, bottoms1, lefts2, tops2, rights2, bottoms2 = df["path"], df["left1"], df["top1"], df["right1"], df["bottom1"], df["left2"], df["top2"], df["right2"], df["bottom2"]
        lefts3, tops3, rights3, bottoms3, lefts4, tops4, rights4, bottoms4 = df["left3"], df["top3"], df["right3"], df["bottom3"], df["left4"], df["top4"], df["right4"], df["bottom4"]
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
        self.lefts3 = lefts3.values
        self.tops3 = tops3.values
        self.rights3 = rights3.values
        self.bottoms3 = bottoms3.values
        self.lefts4 = lefts4.values
        self.tops4 = tops4.values
        self.rights4 = rights4.values
        self.bottoms4 = bottoms4.values
        self.inDim = inDim

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x = Image.open(path)
        left1, top1, right1, bottom1 = self.lefts1[idx], self.tops1[idx], self.rights1[idx], self.bottoms1[idx]
        left2, top2, right2, bottom2 = self.lefts2[idx], self.tops2[idx], self.rights2[idx], self.bottoms2[idx]
        left3, top3, right3, bottom3 = self.lefts3[idx], self.tops3[idx], self.rights3[idx], self.bottoms3[idx]
        left4, top4, right4, bottom4 = self.lefts4[idx], self.tops4[idx], self.rights4[idx], self.bottoms4[idx]

        if self.transform:
            x = self.transform(x)  # xはtorch型

        #x = x.repeat(3, 1, 1)  # grayからRGBの形に変換。(3,300,300)
        x = x.repeat(self.inDim, 1, 1)  # grayからRGBの形に変換。(3,300,300)

        #正常画像が現れる可能性があるので、修正した。
        boxes = []
        if str(left1)!="nan":
            #異常画像
            boxes.append([left1, top1, right1, bottom1])
            if str(left2)!="nan":
                #2個目の病変があるケース
                boxes.append([left2, top2, right2, bottom2])
                if str(left3) != "nan":
                    boxes.append([left3, top3, right3, bottom3])
                    if str(left4) != "nan":
                        boxes.append([left4, top4, right4, bottom4])

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
    df["left3"] = df["left3"] * size / originalSize
    df["top3"] = df["top3"] * size / originalSize
    df["right3"] = df["right3"] * size / originalSize
    df["bottom3"] = df["bottom3"] * size / originalSize
    df["left4"] = df["left4"] * size / originalSize
    df["top4"] = df["top4"] * size / originalSize
    df["right4"] = df["right4"] * size / originalSize
    df["bottom4"] = df["bottom4"] * size / originalSize
    return df

def collate_fn(batch):
    return tuple(zip(*batch))

#バグ修正版。
def visualize(model, dataloader, dataframe, numSamples, saveDir, numDisplay=4, thres=0.4, show_gt=True, show_inf=True, maxsize=None):

    torch.cuda.empty_cache()
    model.eval()

    #少なくともnumSamples以上のデータ数は出力する仕様にする
    num = 0

    for images, targets in dataloader:
        images = list(img.to(device) for img in images)
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

            # firstly, filtering by the top numDisplay
            boxes = boxes[:numDisplay].astype(np.int32)
            scores = scores[:numDisplay]

            # also, we display the bboxes whose score is over threshold
            boxes = boxes[scores >= thres]
            scores = scores[scores >= thres]

            #visualization
            if show_inf==True:
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
            if show_gt==True:
                for i, box in enumerate(trueBoxes):
                    draw = ImageDraw.Draw(image)
                    if maxsize==None:
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="blue", width=3)
                    else:
                        #花岡先生に見せる用。bboxの大きさがmaxsize以下かそうでないかで色を使い分ける。
                        if box[2] <= box[0] + maxsize and box[3] <= box[1] + maxsize:
                            color = "blue"
                        else:
                            color = "green"
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)

            # set the savePath
            fileName = dataframe["file"][num+ind]  #fileName = dataframe["file"][ind]
            savePath = saveDir + fileName

            # 画像を保存したい時用
            #image.save(f"resample_test{str(i)}.png")
            image.save(savePath)

        num += len(images)
        if num >= numSamples:
            break


def train(dataloader, model, optimizer):
    total = 0
    sum_loss = 0

    #training
    model.train()
    for images, bboxes in dataloader:
        images = list(image.to(device) for image in images)
        batch = len(images)

        targets = [{"boxes": bbox.to(device), "labels": torch.ones(len(bbox), dtype=torch.int64).to(device)} for bbox in bboxes]

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


def trainWithCls(dataloader, model, optimizer):
    total = 0
    sum_loss = 0

    #training
    model.train()
    for images, bboxes in dataloader:
        images = list(image.to(device) for image in images)
        batch = len(images)

        targets = [{"boxes": bbox.to(device), "labels": torch.ones(len(bbox), dtype=torch.int64).to(device)} for bbox in bboxes]

        def closure():
            # return from the model
            loss_dict = model(images, targets)  # 返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）
            losses = sum(loss for loss in loss_dict.values())
            #loss = losses.item()

            optimizer.zero_grad()
            losses.backward()
            return losses #loss

        #loss = optimizer.step(closure)
        losses = optimizer.step(closure)
        loss = losses.item()

        total += batch
        sum_loss += loss

    return sum_loss / total


def valid(dataloader, model):
    total = 0
    sum_loss = 0

    # validation
    #model.train() あえてつける必要はないが、結局これと同じ。with torch.no_grad(): をつけている間は誤差逆伝搬はなされないのでOK。
    model.train()
    for images, bboxes in dataloader:
        images = list(image.to(device) for image in images)
        batch = len(images)

        targets = [{"boxes": bbox.to(device), "labels": torch.ones(len(bbox), dtype=torch.int64).to(device)} for bbox in bboxes]

        # return from the model
        loss_dict = model(images, targets)  # 返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）
        losses = sum(loss for loss in loss_dict.values())
        loss = losses.item()

        total += batch
        sum_loss += loss

    return sum_loss / total


def mIoU(dataloader, model, numDisplay=2):

    total = 0
    sum_miou = 0

    model.eval()
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
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


def mAP(dataloader, model, numDisplay=2):

    total = 0
    sum_ap = 0

    iou_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    model.eval()
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
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


def returnFnIndices(dataloader, model, numDisplay, thres=0):

    total = 0
    fn_indices = []
    model.eval()

    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
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

            #deal with when no bbox is predicted. When len(boxes)==0, it should be counted as FN
            if len(boxes)==0:
                fn_indices.append(total + i)
            else: #if len(boxes)>0:
                # calculate iou with targets and outputs
                bboxIou = box_iou(boxes, trueBoxes)
                maxIou = bboxIou.max(axis=0).values.numpy()  #calculate the maximum IoU for each trueBox
                meanIou = maxIou.mean()  #sum(maxIou)/len(maxIou)
                if meanIou <= thres:
                    fn_indices.append(total + i)

        total += batch

    return fn_indices


def returnFnIndices_Dice(dataloader, model, numDisplay, thres=0, lower_thres=-1):

    total = 0
    fn_indices = []
    model.eval()

    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
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

            #deal with when no bbox is predicted. When len(boxes)==0, it should be counted as FN
            if len(boxes)==0:
                if lower_thres < 0: #and 0 <=thres:
                    fn_indices.append(total + i)
            else: #if len(boxes)>0:
                # calculate iou with targets and outputs
                bboxDice = box_dice(boxes, trueBoxes)
                maxDice = bboxDice.max(axis=0).values.numpy()  #calculate the maximum Dice for each trueBox
                meanDice = maxDice.mean()
                if lower_thres < meanDice and meanDice <= thres: #if meanIou <= thres:
                    fn_indices.append(total + i)

        total += batch

    return fn_indices

'''
calculate dice, referring to the pytorch documentation:
https://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html#box_iou
'''

def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def box_area(boxes):
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def _box_inter_union(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_dice(boxes1, boxes2):
    inter, union = _box_inter_union(boxes1, boxes2)
    dice = 2*inter / (inter + union)
    return dice


def mDice(dataloader, model, numDisplay=4, thres=0.4):

    total = 0
    sum_dice = 0

    model.eval()
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        batch = len(images)
        outputs = model(images)

        #calculate dice with targets and outputs
        for i in range(batch):
            trueBoxes = targets[i]
            boxes = outputs[i]["boxes"].data.cpu()  #.numpy()
            scores = outputs[i]["scores"].data.cpu()  #.numpy()

            # firstly, filtering by the top numDisplay
            boxes = boxes[:numDisplay]
            scores = scores[:numDisplay]

            # also, we display the bboxes whose score is over threshold
            boxes = boxes[scores >= thres]
            scores = scores[scores >= thres]

            #deal with when no bbox is predicted. When len(boxes)==0, sum_dice += 0
            if len(boxes)>0:
                # calculate dice with targets and outputs
                bboxDice = box_dice(boxes, trueBoxes)
                maxDice = bboxDice.max(axis=0).values.numpy()  #calculate the maximum Dice for each trueBox
                meanDice = maxDice.mean()
                sum_dice += meanDice

        total += batch

    return sum_dice / total

'''
以下、FROC用
'''

def isSmall(box, size_thres):
    #if the size of the box is <= size_thres, then return True
    return box[2] <= box[0] + size_thres and box[3] <= box[1] + size_thres

def evaluate(trueBoxes, boxes, hit_thres, size_thres, ignore_big_bbox, accept_TP_duplicate):
    # default: hit_thres=0.2, ignore_big_bbox=False, accept_TP_duplicate=True
    # 場合分けを各々やればよい

    bboxDice = box_dice(boxes, trueBoxes)
    TP, FP = 0, 0
    gt_used = [0] * len(trueBoxes)  # non-used yet.

    if ignore_big_bbox==False and accept_TP_duplicate==True: #一番ナイーブ
        maxDice = bboxDice.max(axis=1).values.numpy() #calculate the maximum Dice for each bbox, not trueBox
        gt_indices = bboxDice.argmax(axis=1)
        for i,dice in enumerate(maxDice):
            if dice >= hit_thres:
                index = gt_indices[i]
                gt_used[index] = 1 #assigned
        TP = sum(gt_used)
        FP = sum(maxDice < hit_thres)


    elif ignore_big_bbox == False and accept_TP_duplicate == False:
        # 重複を許さないので、こっちの方が厳しめの判定にはなる。
        # scoreの大きい予測bboxから順に、まだ未割当の正解bboxとのマッチングを、greedyに行っていけばよい。

        sortedDice = bboxDice.sort(axis=1, descending=True)
        sortedValues = sortedDice.values
        sortedIndices = sortedDice.indices

        # scoreが高い順のgreedyなので、上から順にサーベイしていけばよい
        for i in range(len(boxes)):  # len(boxes) should be equal to len(sortedValues)

            # 今、着目しているggt_bboxのindexはsortedIndices[i][j]であることに留意。
            j = 0
            while sortedValues[i][j] >= hit_thres and gt_used[sortedIndices[i][j]] == 1:
                j += 1
                if j == len(trueBoxes):  # len(trueBoxes) should be equal to len(sortedValues[0])
                    break

            if j == len(trueBoxes):
                # 常にhitしていたのだが、マッチするgt_bboxはむなしく結局なかった。
                FP += 1
            elif sortedValues[i][j] < hit_thres:
                FP += 1
            else:  # the gt_bbox is not assigned yet  #sortedValues[i][j] >= hit_thres holds.
                TP += 1
                gt_used[sortedIndices[i][j]] = 1  # assign

    else:
        #elif ignore_big_bbox == True and accept_TP_duplicate == True:
        #Or, ignore_big_bbox == True and accept_TP_duplicate == False
        # 大bboxは無視しして扱う＆TPの重複を許すversion
        # 結局、argmaxだけ考えることにした。２番手以降だとdice>=0.2が発生することはほとんどないだろうから。

        maxDice = bboxDice.max(axis=1).values.numpy() #calculate the maximum Dice for each bbox, not trueBox
        gt_indices = bboxDice.argmax(axis=1)
        for i, dice in enumerate(maxDice):
            if dice >= hit_thres:
                index = gt_indices[i]
                if isSmall(trueBoxes[index], size_thres): #small bbox
                    if gt_used[index] == 0: #not assigned yet
                        TP += 1
                        gt_used[index] = 1
                    else:
                        #already assigned
                        #when accept_TP_duplicate, just ignore
                        if not accept_TP_duplicate:
                            FP += 1
                # else:
                    #big bbox
                    #大きなbboxとマッチした予測bboxはIPとして扱え。すなわちIgnore。
            else:
                FP += 1

    return TP, FP


def FROC(dataloader, model, hit_thres=0.2, size_thres=150 * 300 / 1024, thresholds=[0.3, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1], ignore_big_bbox=False, accept_TP_duplicate=True):

    #thresholds = [0.3, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]  # 仮に。もっと細かくしてもよい。
    numThres = len(thresholds)

    numTP = np.zeros(numThres)  # [0] * numThres
    numFP = np.zeros(numThres)  # [0] * numThres

    numGT = 0  # gt_bboxの数. 各thresに対して同一になるはずなので、一回計算すればよい。
    total = 0

    model.eval()
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        batch = len(images)
        outputs = model(images)

        for ind in range(batch):
            trueBoxes = targets[ind]
            boxes = outputs[ind]["boxes"].data.cpu()  # .numpy()
            scores = outputs[ind]["scores"].data.cpu()  # .numpy()

            #count the number of ground truth
            if ignore_big_bbox == True:
                for box in trueBoxes:
                    if isSmall(box, size_thres):  # bboxが小さいならGTの中に加えよ
                        numGT += 1
            else:
                numGT += len(trueBoxes)

                
            #count the number of TP and FP in this image for each numThres
            for i in range(numThres):
                ###naive implementation is the first step
                boxes = boxes[scores >= thresholds[i]]
                scores = scores[scores >= thresholds[i]]

                # case distinction
                if len(trueBoxes)==0: #normal image
                    TP = 0
                    FP = len(boxes)
                else:
                    #以下、関数化する。boxesとtrueBoxesをもとに、TPとFPをcalculateするモジュール。
                    TP, FP = evaluate(trueBoxes, boxes, hit_thres, size_thres, ignore_big_bbox, accept_TP_duplicate)

                numTP[i] += TP
                numFP[i] += FP

        total += batch

    TPRs = numTP / numGT
    FPIs = numFP / total

    return TPRs, FPIs, thresholds  # numTP, numFP, numGT, total, thresholds


def interpolate(TPRs, FPIs, x):
    # based on the input, return the value at the point x, i.e., FPIs(x)
    # ついでにそのindex iも返そう（その方が便利）
    # Both is fine, reversed or not

    # Preprocessing: the input should be an ascending order
    if TPRs[0] > TPRs[-1]:
        TPRs = list(reversed(TPRs))
        FPIs = list(reversed(FPIs))

    # find the index
    # ここでエラー発生したら、それはextrapolationになってしまうので、エラーたるべき。
    i = 0
    while not (FPIs[i] <= x and x <= FPIs[i + 1]):
        i += 1

    # interpolated value at FPIs(x)
    x1, x2 = FPIs[i], FPIs[i + 1]
    y1, y2 = TPRs[i], TPRs[i + 1]
    y = ((x2 - x) * y1 + (x - x1) * y2) / (x2 - x1)
    return y, i


def FAUC(TPRs, FPIs):
    # calculate FAUC i.e., the area of FROC from 0 to 1
    # Notice: the input array are reversed. so we have to reverse, or return its abs()

    # Preprocessing: make the input an ascending order
    if TPRs[0] > TPRs[-1]:
        TPRs = list(reversed(TPRs))
        FPIs = list(reversed(FPIs))

    #例外処理
    if FPIs[-1] < 1:
        return 0

    #else:
    # interpolate
    y, index = interpolate(TPRs, FPIs, 1)

    # add the element at the point 1 to the appropriate index
    FPIs[index + 1] = 1
    TPRs[index + 1] = y

    # delete the area outside of 1, i.e., 1 < x
    FPIs = FPIs[:index + 2]
    TPRs = TPRs[:index + 2]

    return integrate.trapz(TPRs, FPIs)  # (y, x)


def CPM(TPRs, FPIs):
    # calculate an average sensitivity at the following 7 points
    # Notice: the input array are reversed. so we have to reverse, or return its abs()

    xs = [1/8, 1/4, 1/2, 1, 2, 4, 8]
    sums = 0

    # Preprocessing: make the input an ascending order
    if TPRs[0] > TPRs[-1]:
        TPRs = list(reversed(TPRs))
        FPIs = list(reversed(FPIs))

    #例外処理
    if FPIs[-1] < 8:
        return 0

    for x in xs:
        # interpolate
        y, _ = interpolate(TPRs, FPIs, x)
        sums += y

    return sums/len(xs)


def RCPM(TPRs, FPIs):
    # calculate an average sensitivity at the following 7 points
    # Notice: the input array are reversed. so we have to reverse, or return its abs()

    xs = [1/8, 1/4, 1/2, 1]
    sums = 0

    # Preprocessing: make the input an ascending order
    if TPRs[0] > TPRs[-1]:
        TPRs = list(reversed(TPRs))
        FPIs = list(reversed(FPIs))

    #例外処理
    if FPIs[-1] < 1:
        return 0

    for x in xs:
        # interpolate
        y, _ = interpolate(TPRs, FPIs, x)
        sums += y

    return sums/len(xs)


def plotFROC(TPRs, FPIs, savePath, include_FPIs=8):
    #include_FPIs: 少なくとも含むべきxの範囲。すなわち、これ以上のところは可視化する際にはカットする。

    # Preprocessing: make the input an ascending order
    if TPRs[0] > TPRs[-1]:
        TPRs = list(reversed(TPRs))
        FPIs = list(reversed(FPIs))

    #case distinction for visualization
    if include_FPIs < FPIs[-1]:

        # interpolate
        _, index = interpolate(TPRs, FPIs, include_FPIs)

        # delete the area outside of include_FPIs
        FPIs = FPIs[:index + 2]
        TPRs = TPRs[:index + 2]

    plt.figure()
    plt.plot(FPIs, TPRs, 'o-')  #点を表示させないなら、plt.plot(FPIs, TPRs)でよい。
    plt.xlabel('FPavg')
    plt.ylabel('Sensitivity')

    plt.savefig(savePath)

