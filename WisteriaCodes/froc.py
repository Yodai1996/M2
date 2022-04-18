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
from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, mDice, FROC, FAUC, CPM, RCPM, plotFROC, modifyArea

'''
just for making a result
'''

args = sys.argv
#Model Path
saveFROCPath, end = args[1], float(args[2])
UDRPath, BayRnPath, CDRPath, easy2hard1Path, easy2hard2Path = args[3], args[4], args[5], args[6], args[7]
i = int(args[8])

# hypara
originalSize = 1024
size = 300
batch_size=64 #ただのinferenceなので、batch sizeは何でもよい
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

#linestyles=[(0, (1, 0)), (0, (1, 1)), (0, (5, 1)), (0, (5, 1, 1, 1)),  ]
plt.figure()


#for i in range(5):
bbox = "rare_small_bboxInfo_81_{}_withNormal.csv".format(str(i))
dataPath = 'AllDataDir'
dataDir = "/work/gk36/k77012/M2/data/{}/".format(dataPath)
df = pd.read_csv("/work/gk36/k77012/M2/{}".format(bbox))

df = preprocess_df(df, originalSize, size, dataDir)
dataset = MyDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1

#UDR
UDRmodel = models.detection.ssd300_vgg16().to(device)
model_head = models.detection.ssd300_vgg16(num_classes=num_classes)
UDRmodel.head.classification_head = model_head.head.classification_head.to(device)  # modify the head of the model
UDRmodel.load_state_dict(torch.load(UDRPath))

#BayRn
Baymodel = models.detection.ssd300_vgg16().to(device)
model_head = models.detection.ssd300_vgg16(num_classes=num_classes)
Baymodel.head.classification_head = model_head.head.classification_head.to(device)  # modify the head of the model
Baymodel.load_state_dict(torch.load(BayRnPath))

#CDR
CDRmodel = models.detection.ssd300_vgg16().to(device)
model_head = models.detection.ssd300_vgg16(num_classes=num_classes)
CDRmodel.head.classification_head = model_head.head.classification_head.to(device)  # modify the head of the model
CDRmodel.load_state_dict(torch.load(CDRPath))

#easy2hard1
easy2hard1model = models.detection.ssd300_vgg16().to(device)
model_head = models.detection.ssd300_vgg16(num_classes=num_classes)
easy2hard1model.head.classification_head = model_head.head.classification_head.to(device)  # modify the head of the model
easy2hard1model.load_state_dict(torch.load(easy2hard1Path))

#easy2hard2
easy2hard2model = models.detection.ssd300_vgg16().to(device)
model_head = models.detection.ssd300_vgg16(num_classes=num_classes)
easy2hard2model.head.classification_head = model_head.head.classification_head.to(device)  # modify the head of the model
easy2hard2model.load_state_dict(torch.load(easy2hard2Path))

# UDR
TPRs, FPIs, thresholds = FROC(dataloader, UDRmodel, ignore_big_bbox=True)
TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
plt.plot(FPIs, TPRs, label="UDR (Adam)", linestyle=(0, (1, 0)))

# BayRn
TPRs, FPIs, thresholds = FROC(dataloader, Baymodel, ignore_big_bbox=True)
TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
plt.plot(FPIs, TPRs, label="BayRn (Adam)", linestyle=(0, (1, 1)))

# CDR
TPRs, FPIs, thresholds = FROC(dataloader, CDRmodel, ignore_big_bbox=True)
TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
plt.plot(FPIs, TPRs, label="GDR (NVRM-SGD)", linestyle=(0, (3, 1)))

#e2h1
TPRs, FPIs, thresholds = FROC(dataloader, easy2hard1model, ignore_big_bbox=True)
TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
plt.plot(FPIs, TPRs, label="Easy2Hard-1 (NVRM-SGD)", linestyle=(0, (5, 1, 1, 1)))

#e2h2
TPRs, FPIs, thresholds = FROC(dataloader, easy2hard2model, ignore_big_bbox=True)
TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
plt.plot(FPIs, TPRs, label="Easy2Hard-2 (NVRM-SGD)", linestyle=(0, (8, 1)))

plt.xlabel('FPs/Image')
plt.ylabel('TPR')
plt.legend()
plt.savefig(saveFROCPath + f"FROC_fold{i}_end{end}.png")


