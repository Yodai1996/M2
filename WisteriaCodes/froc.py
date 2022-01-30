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
saveFROCPath, end, nonrarePath = args[1], float(args[2]), args[3]
BayRnPaths = args[4], args[5], args[6], args[7], args[8]
CDRPaths = args[9], args[10], args[11], args[12], args[13]

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

for i in range(5):
    bbox = "rare_small_bboxInfo_81_{}_withNormal.csv".format(str(i+1))
    dataPath = 'AllDataDir'
    dataDir = "/work/gk36/k77012/M2/data/{}/".format(dataPath)
    df = pd.read_csv("/work/gk36/k77012/M2/{}".format(bbox))

    df = preprocess_df(df, originalSize, size, dataDir)
    dataset = MyDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

    #load model
    BayRnPath = BayRnPaths[i]
    CDRPath = CDRPaths[i]

    num_classes = 2 #(len(classes)) + 1

    #nonrare
    nonraremodel = models.detection.ssd300_vgg16().to(device)
    model_head = models.detection.ssd300_vgg16(num_classes=num_classes)
    nonraremodel.head.classification_head = model_head.head.classification_head.to(device)  # modify the head of the model
    nonraremodel.load_state_dict(torch.load(nonrarePath))

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

    #inference
    #nonrare
    TPRs, FPIs, thresholds = FROC(dataloader, nonraremodel, ignore_big_bbox=True)
    TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
    if i==0:
        plt.plot(FPIs, TPRs, "g", label="nonrare pneumonia dataset", linestyle=":")
    else:
        plt.plot(FPIs, TPRs, "g", linestyle=":")

    # BayRn
    TPRs, FPIs, thresholds = FROC(dataloader, Baymodel, ignore_big_bbox=True)
    TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
    if i == 0:
        plt.plot(FPIs, TPRs, "b", label="BayRn", linestyle="--")
    else:
        plt.plot(FPIs, TPRs, "b", linestyle="--")

    # CDR
    TPRs, FPIs, thresholds = FROC(dataloader, CDRmodel, ignore_big_bbox=True)
    TPRs, FPIs = modifyArea(TPRs, FPIs, include_FPIs=end)
    if i == 0:
        plt.plot(FPIs, TPRs, "r", label="CDR", linestyle="-")
    else:
        plt.plot(FPIs, TPRs, "r", linestyle="-")


plt.xlabel('FPs/Image')
plt.ylabel('TPR')
plt.legend()
plt.savefig(saveFROCPath + f"FROC_end{end}.png")