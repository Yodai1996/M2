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
from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP

'''
just for inference
'''

args = sys.argv
version, iter, abnormalDir, bboxPath, boText = int(args[1]), int(args[2]), args[3], args[4], args[5]

valueList = [[0.6, 0.4166666666666667, 0.4, 0.49999999999999994, 0.59, 0.4444444444444445, 0.3333333333333333],
             [0.8, 0.6666666666666666, 0.7, 0.37499999999999994, 0.45, 0.2222222222222222, 0.8333333333333331],
             [0.4, 0.75, 0.6, 0.37499999999999994, 0.5, 0.5555555555555556, 0.4999999999999999],
             [0.999, 0.8333333333333334, 0.6666666666666666, 0.12499999999999997, 0.9, 0.99, 0.01],
             [0.2, 0.25, 0.9, 0.6249999999999999, 0.1, 0.6666666666666666, 0.6666666666666665]]

values = valueList[iter-1] #from 0 to 4

# preprocess
lb = int(30 + 50 * values[0])  #[30,80]
ub = int(100 + 120 * values[1]) #[100,220]
res = int(2 + 4 * values[2])
octaves = 5 #fixed
persistence = 0.2 + 0.8 * values[3] #[0.2,1]
lacunarity = int(2 + 3 * values[4])
scale = 0.1 + 0.9 * values[5] #[0.1,1]
smoothArea = 0.2 + 0.6 * values[6] #[0.2,0.8]

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
    PATH = "/lustre/gk36/k77012/M2/model/adversarialBO/model{}".format(version-1)  #version-1 represents the previous step
    model.load_state_dict(torch.load(PATH))

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

with torch.no_grad():
    miou = mIoU(dataloader, model, numDisplay)

# save the values and score for the next iteration
values = [str(v) for v in values]
values = ", ".join(values)

fileHandle = open(boText, "a")
fileHandle.write(values + ", " + str(miou*(-1)) + "\n")  ###adversarialBOなので、マイナスをつけたものを最大化する。
fileHandle.close()
