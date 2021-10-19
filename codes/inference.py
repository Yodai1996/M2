import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image, ImageDraw, ImageFilter
import scipy.sparse as sp
import pandas as pd
from numpy.random import *

import torch
from torch import nn, optim
import torch.nn.functional as F
import copy

from torchvision import models, transforms, datasets
from skimage.transform import resize

from skimage import io, transform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, mDice

'''
just for inference
'''

args = sys.argv
version, bufText, abnormalDir, bboxPath, boText, modelPath, metric = int(args[1]), args[2], args[3], args[4], args[5], args[6], args[7]

# read the recommended next values from Gaussian Process.
fileHandle = open(bufText, "r")
lineList = fileHandle.readlines()
fileHandle.close()
last_lines = lineList[-1]

if last_lines[-1] == "\n":
    last_lines = last_lines[:-2]

values = last_lines.split(",")
values = [float(i) for i in values]

# preprocess
lb = int(30 + 50 * values[0])  # [30,80]
ub = int(100 + 120 * values[1])  # [100,220]
res = int(2 + 4 * values[2])
octaves = 5  # fixed
persistence = 0.2 + 0.8 * values[3]  # [0.2,1]
lacunarity = int(2 + 3 * values[4])
scale = 0.1 + 0.9 * values[5]  # [0.1,1]
smoothArea = 0.2 + 0.6 * values[6]  # [0.2,0.8]

print(values)
print(lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)

# hypara
originalSize = 1024
size = 300
lr = 0.0002
batch_size=16 #ただのinferenceなので、batch sizeは何でもよい
modelName="SSD" #まあSSDでいいでしょ
pretrained="pretrained"
numDisplay = 2 #the number of predicted bboxes to display, also used when calculating mIoU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

df = pd.read_csv(bboxPath)
df = preprocess_df(df, originalSize, size, abnormalDir)
dataset = MyDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1
if modelName=="SSD":
    model = models.detection.ssd300_vgg16(pretrained=False, pretrained_backbone=False).to(device)
    if pretrained=="pretrained":
        model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/ssd300_vgg16_coco-b556d3b4.pth"))
    model2 = models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)
    model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model
else:  #modelName=="fasterRCNN"
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
    if pretrained == "pretrained":
        model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))  # model.load_state_dict(torch.load("/lustre/gk36/k77012/faster_RCNN.pth"))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

#load the previously trained model
if version >= 2:
    PATH = modelPath + "model" + str(version-1)
    model.load_state_dict(torch.load(PATH))

with torch.no_grad():
    if metric == "IoU":
        infer = mIoU(dataloader, model, numDisplay)
    else:  # metric=="Dice"
        infer = mDice(dataloader, model, numDisplay)

fileHandle = open(boText, "a")
fileHandle.write(last_lines + ", " + str(infer*(-1)) + "\n")  ###adversarialBOなので、マイナスをつけたものを最大化する。
fileHandle.close()
