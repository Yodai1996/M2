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
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def visualize(model, dataloader, dataframe, numSamples, saveDir, numDisplay=2):

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

        #     fig, ax = plt.subplots(1, 1)
        #     ax.imshow(np.array(image))
        #
        # plt.show()


#上位numDisplay個を表示する仕様にする.
#next(iter(dataloader))では期待通りには動かない。バグあり。
#def visualize(model, dataset, dataframe, originalSize, size, numSamples, saveDir, numDisplay=2):
# def visualize(model, dataloader, dataframe, numSamples, saveDir, numDisplay=2):
#
#     torch.cuda.empty_cache()
#     model.eval()
#
#     #少なくともnumSamples以上のデータ数は出力する仕様にする
#     num = 0
#     while num < numSamples:
#
#         images, targets = next(iter(dataloader))
#         images = list(img.to(device) for img in images)
#
#         outputs = model(images)
#
#         for ind, image in enumerate(images):
#             #image shape is (3,300,300)
#             image = image.permute(1, 2, 0).cpu().numpy() #bboxを白黒表示したいなら、この代わりに、image = image[0].cpu().numpy() #(300,300) でよい
#             image = Image.fromarray((image * 255).astype(np.uint8))
#             trueBoxes = targets[ind]
#
#             #outputs
#             boxes = outputs[ind]["boxes"].data.cpu().numpy()
#             scores = outputs[ind]["scores"].data.cpu().numpy()
#             #labels = outputs[ind]["labels"].data.cpu().numpy()
#
#             # filtering by the top numDisplay
#             boxes = boxes[:numDisplay].astype(np.int32)
#             scores = scores[:numDisplay]
#
#             #visualization
#             for i, box in enumerate(boxes):
#                 draw = ImageDraw.Draw(image)
#                 score = scores[i]
#                 label = "score:{:.2f}".format(score)
#                 draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
#
#                 #fnt = ImageFont.truetype('/content/mplus-1c-black.ttf', 20)
#                 #fnt = ImageFont.truetype("arial.ttf", 10) #size
#                 fnt = ImageFont.load_default()
#                 text_w, text_h = fnt.getsize(label)
#                 draw.rectangle([box[0], box[1], box[0]+text_w, box[1]+text_h], fill="red")
#                 draw.text((box[0], box[1]), label, font=fnt, fill='white')
#
#             #visualization
#             for i, box in enumerate(trueBoxes):
#                 draw = ImageDraw.Draw(image)
#                 draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="blue", width=3)
#
#             # set the savePath
#             fileName = dataframe["file"][num+ind]  #fileName = dataframe["file"][ind]
#             savePath = saveDir + fileName
#
#             # 画像を保存したい時用
#             #image.save(f"resample_test{str(i)}.png")
#             image.save(savePath)
#
#         num += len(images)
#
#         #     fig, ax = plt.subplots(1, 1)
#         #     ax.imshow(np.array(image))
#         #
#         # plt.show()

#元のvisualizaで使われていたGenerateBboxやshow_corner_bb_trueBbox.ちゃんと動くよ
# def GenerateBbox(box, color="red"):
#     left, top, right, bottom = box[0], box[1], box[2], box[3]
#     return plt.Rectangle((left, top), right - left, bottom - top, color=color, fill=False, lw=2)
#
# def show_corner_bb_trueBbox(x, boxes, trueBoxes=None, savePath=None):
#     #trueBoxesはboxesと同じ型。
#
#     plt.imshow(x)
#
#     #かなりださいので要修正かな.
#     #最大表示数3つとしている.
#     for i, box in enumerate(boxes):
#         if i==0:
#             bbox1 = GenerateBbox(box)
#             plt.gca().add_patch(bbox1)
#         elif i==1:
#             bbox2 = GenerateBbox(box)
#             plt.gca().add_patch(bbox2)
#         elif i ==2:
#             bbox3 = GenerateBbox(box)
#             plt.gca().add_patch(bbox3)
#         else:
#             break
#
#     for i, box in enumerate(trueBoxes):
#         if i==0:
#             trueBbox1 = GenerateBbox(box, color="blue")
#             plt.gca().add_patch(trueBbox1)
#         elif i==1:
#             trueBbox2 = GenerateBbox(box, color="blue")
#             plt.gca().add_patch(trueBbox2)
#         elif i ==2:
#             trueBbox3 = GenerateBbox(box, color="blue")
#             plt.gca().add_patch(trueBbox3)
#         else:
#             break
#
#     if savePath!=None:
#         plt.savefig(savePath)
#
#     #次の画像に影響与えうるので、removeしておく。
#     for i in range(len(boxes)):
#         if i==0:
#             bbox1.remove()
#         elif i==1:
#             bbox2.remove()
#         elif i ==2:
#             bbox3.remove()
#         else:
#             break
#
#     #次の画像に影響与えうるので、removeしておく。
#     for i in range(len(trueBoxes)):
#         if i==0:
#             trueBbox1.remove()
#         elif i==1:
#             trueBbox2.remove()
#         elif i ==2:
#             trueBbox3.remove()
#         else:
#             break

#元のvisualiza.ちゃんと動くよ
# def visualize(model, dataset, dataframe, originalSize, size, scoreThres, numSamples, saveDir):
#
#     model.eval()
#     for ind in range(numSamples):
#         x, trueBoxes = dataset[ind]  # (3,300,300) #datasetを直接用いる.
#         x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])  # (1,3,300,300)
#         x = list(x.to(device))  # cuda
#
#         outputs = model(x)
#         boxes = outputs[0]["boxes"].data.cpu().numpy()
#         scores = outputs[0]["scores"].data.cpu().numpy()
#         labels = outputs[0]["labels"].data.cpu().numpy()
#
#         # bboxのサイズを元に戻す。(300➡1024のスケールに。)
#         trueBoxes = trueBoxes * originalSize / size
#         boxes = boxes * originalSize / size
#
#         # filtering by the thres score
#         boxes = boxes[scores >= scoreThres].astype(np.int32)
#         scores = scores[scores >= scoreThres]
#
#         # open the original image
#         testPath = dataframe["path"][ind]
#         x_original = Image.open(testPath)  # (1024,1024)
#
#         # set the savePath
#         fileName = dataframe["file"][ind]
#         savePath = saveDir + fileName
#
#         # start the inference and save the results
#         show_corner_bb_trueBbox(x_original, boxes, trueBoxes=trueBoxes, savePath=savePath)


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


def valid(dataloader, model):
    total = 0
    sum_loss = 0

    # validation
    #model.train() あえてつける必要はないが、結局これと同じ。with torch.no_grad(): をつけている間は誤差逆伝搬はなされないのでOK。
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
