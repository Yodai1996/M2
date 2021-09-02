import sys
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageDraw, ImageFilter

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import models, transforms, datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class MyDataset(Dataset):
    def __init__(self, paths, lefts, tops, rights, bottoms, transform=None):
        self.transform = transform
        self.paths = paths.values
        self.lefts = lefts.values
        self.tops = tops.values
        self.rights = rights.values
        self.bottoms = bottoms.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x = Image.open(path)
        left, top, right, bottom = self.lefts[idx], self.tops[idx], self.rights[idx], self.bottoms[idx]

        if self.transform:
            x = self.transform(x)  # xはtorch型

        x = x.repeat(3, 1, 1)  # grayからRGBの形に変換。(3,300,300)

        bbox = (left, top, right, bottom)
        return x, torch.tensor(bbox, dtype=torch.float)  # ここでreshapeしても良かったか.

def GenerateBbox(box):
    left, top, right, bottom = box[0], box[1], box[2], box[3]
    return plt.Rectangle((left, top), right - left, bottom - top, color="red", fill=False, lw=2)

def show_corner_bb_trueBbox(x, boxes, trueBoxes=None, savePath=None):
    plt.imshow(x)

    #かなりださいので要修正かな.
    #最大表示数3つとしている.
    for i, box in enumerate(boxes):
        if i==0:
            bbox1 = GenerateBbox(box)
            plt.gca().add_patch(bbox1)
        elif i==1:
            bbox2 = GenerateBbox(box)
            plt.gca().add_patch(bbox2)
        elif i ==2:
            bbox3 = GenerateBbox(box)
            plt.gca().add_patch(bbox3)
        else:
            break

    if trueBoxes!=None:
        # add the right answer
        left, top, right, bottom = trueBoxes[0], trueBoxes[1], trueBoxes[2], trueBoxes[3]
        trueBbox = plt.Rectangle((left, top), right - left, bottom - top, color="blue", fill=False, lw=2)
        plt.gca().add_patch(trueBbox)

    if savePath!=None:
        plt.savefig(savePath)

    #次の画像に影響与えうるので、removeしておく。
    for i, box in enumerate(boxes):
        if i==0:
            bbox1.remove()
        elif i==1:
            bbox2.remove()
        elif i ==2:
            bbox3.remove()
        else:
            break

    if trueBoxes != None:
        trueBbox.remove()


# used data
args = sys.argv

version = int(args[1])
trainDir = "/lustre/gk36/k77012/M2/data/train_{}/1_abnormal1000_{}/".format(version, version)
inferDir = "/lustre/gk36/k77012/M2/results/train_{}/".format(version)
df = pd.read_csv("/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/bboxInfo_{}.csv".format(version))

# hypara
originalSize = 1024
size = 300
batch_size = 16
num_epoch = 30
lr = 0.0003
scoreThres = 0.5  # used in inference, you can set as you like.
numSamples = 50 # 50個くらい保存しておくか.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df["path"] = trainDir + df["file"]
# modify the bbox as we resized the input image from originalSize to the size
df["left"] = df["left"] * size / originalSize
df["top"] = df["top"] * size / originalSize
df["right"] = df["right"] * size / originalSize
df["bottom"] = df["bottom"] * size / originalSize

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])
trainset = MyDataset(df["path"], df["left"], df["top"], df["right"], df["bottom"], transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")) # model.load_state_dict(torch.load("/lustre/gk36/k77012/faster_RCNN.pth"))

num_classes = 2  # (len(classes)) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    model.train()
    total = 0
    sum_loss = 0

    for images, bboxes in trainloader:
        batch = images.shape[0]
        images= images.to(device)
        images= list(images) # as the input of faster-RCNN should be the list of the tensors

        # make the list for targets
        targets = [{"boxes": bbox.reshape(1, 4).to(device), "labels": torch.ones(1, dtype=torch.int64).to(device)} for bbox in bboxes]

        # return from the model
        loss_dict = model(images, targets)  # 返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）
        losses = sum(loss for loss in loss_dict.values())
        loss = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total += batch
        sum_loss += loss

    train_loss = sum_loss / total
    print("epoch:{}/{}  train_loss:{:.3f}".format(epoch + 1, num_epoch, train_loss))

# visualization
for ind in range(numSamples):
    x, trueBoxes = trainset[ind]  # (3,300,300) #datasetを直接用いる.
    x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])  # (1,3,300,300)
    x = list(x.to(device)) #cuda

    model.eval()
    outputs = model(x)
    boxes = outputs[0]["boxes"].data.cpu().numpy()
    scores = outputs[0]["scores"].data.cpu().numpy()
    labels = outputs[0]["labels"].data.cpu().numpy()

    # bboxのサイズを元に戻す。(300➡1024のスケールに。)
    trueBoxes = trueBoxes * originalSize / size
    boxes = boxes * originalSize / size

    #filtering by the thres score
    boxes = boxes[scores >= scoreThres].astype(np.int32)
    scores = scores[scores >= scoreThres]

    # open the original image
    testPath = df["path"][ind]
    x_original = Image.open(testPath)  # (1024,1024)

    # set the savePath
    fileName = df["file"][ind]
    savePath = inferDir + fileName

    # start the inference and save the results
    show_corner_bb_trueBbox(x_original, boxes, trueBoxes=trueBoxes, savePath=savePath)