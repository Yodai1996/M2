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

from fractalGenerator import make_abnormals
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

from skimage import io, transform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, mDice, FROC, FAUC, CPM, RCPM, plotFROC
'''
just for inference
'''

args = sys.argv
version, iter, abnormalDir, bboxPath, boText, modelPath, pretrained = int(args[1]), int(args[2]), args[3], args[4], args[5], args[6], args[7]
if len(args)>8: #curriculumBO
    start, decayRate = float(args[8]), float(args[9])

valueList = [[0.4, 0.6249999999999999, 0.55, 0.5555555555555556, 0.4999999999999999],
             [0.6, 0.12499999999999997, 0.8, 0.7777777777777778, 0.1666666666666666],
             [0.7, 0.37499999999999994, 0.45, 0.2222222222222222, 0.8333333333333331],
             [0.95, 0.25, 0.15, 0.6666666666666666, 0.6666666666666665],
             [0.1, 0.7500000000000001, 0.9, 0.4444444444444445, 0.3333333333333333]]

values = valueList[iter-1] #from 0 to 4

# preprocess
# fix these for small bbox detection
lb = 20 #int(30 + 50 * values[0])  #[30,80]
ub = 75 #int(100 + 120 * values[1]) #[100,220]
octaves = 5 #fixed
res = int(2 + 4 * values[0])
persistence = 0.2 + 0.8 * values[1] #[0.2,1]
lacunarity = int(2 + 3 * values[2])
scale = 0.1 + 0.9 * values[3] #[0.1,1]
smoothArea = 0.2 + 0.6 * values[4] #[0.2,0.8]

print(values)
print(lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)

# hypara
originalSize = 1024
size = 300
lr = 0.0002
batch_size=16 #ただのinferenceなので、batch sizeは何でもよい
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
modelName="SSD" #まあSSDでいいでしょ
if modelName=="SSD":
    if pretrained=="pretrained" or pretrained=="ImageNet": #same meaning
        model = models.detection.ssd300_vgg16(pretrained=True).to(device)
    else:
        model = models.detection.ssd300_vgg16(pretrained=False).to(device)
    model2 = models.detection.ssd300_vgg16(num_classes = num_classes) #models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)
    model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model

    if pretrained=="BigBbox":
        loadModelPath="/work/gk36/k77012/M2/model/pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120" #とりあえず、VSGD, noise=0.01を使用すれことにする。
        model.load_state_dict(torch.load(loadModelPath))

#load the previously trained model
if version >= 2:
    PATH = modelPath + "model" + str(version-1)
    model.load_state_dict(torch.load(PATH))

#inference
thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]
TPRs, FPIs, thresholds = FROC(dataloader, model, thresholds=thresholds, ignore_big_bbox=True)
infer = FAUC(TPRs, FPIs)

if len(args) > 8:  # curriculumBO
    if decayRate == 0 or decayRate == 1:
        target = start
    elif decayRate < 1:  # else #バグらせるのが怖いので限定的にする。いやしない。
        target = start - decayRate * (version - 1)
    # target = start * (decayRate**version)    #target value in the current curriculum
    obj_function = -1 * abs(infer - target)  #objective function for this min-max formulation
# else: #adversarialBO
#     obj_function = -1 * infer #as it is adversarialBO, we maximize the value multiplied by -1.

# save the values and score for the next iteration
values = [str(v) for v in values]
values = ", ".join(values)

fileHandle = open(boText, "a")
fileHandle.write(values + ", " + str(obj_function) + "\n")
fileHandle.close()