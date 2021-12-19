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
just for making a result
'''

args = sys.argv
dataPath, dataBboxName, Path, loadModelPath, saveDir, saveFROCPath = args[1], args[2], args[3], args[4], args[5], args[6]

bbox = dataBboxName + ".csv"
dataDir = "/work/gk36/k77012/M2/data/{}/".format(dataPath)
df = pd.read_csv("/work/gk36/k77012/M2/{}".format(bbox))

# hypara
originalSize = 1024
size = 300
batch_size=64 #ただのinferenceなので、batch sizeは何でもよい
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

df = preprocess_df(df, originalSize, size, dataDir)
dataset = MyDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1
modelName="SSD" #まあSSDでいいでしょ
if modelName=="SSD":
    if Path[:8]=="ImageNet":
        model = models.detection.ssd300_vgg16(pretrained=True).to(device)
        model2 = models.detection.ssd300_vgg16(num_classes=num_classes)
        model.head.classification_head = model2.head.classification_head.to(device)  # modify the head of the model
    else:
        model = models.detection.ssd300_vgg16(pretrained=False).to(device)
        model2 = models.detection.ssd300_vgg16(num_classes = num_classes)
        model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model
        model.load_state_dict(torch.load(loadModelPath))


#inference
TPRs, FPIs, thresholds = FROC(dataloader, model, ignore_big_bbox=True)
fauc = FAUC(TPRs, FPIs)
cpm = CPM(TPRs, FPIs)
rcpm = RCPM(TPRs, FPIs)
print("test_fauc:{:.4f}  test_cpm:{:.4f}  test_rcpm:{:.4f}".format(fauc, cpm, rcpm))
plotFROC(TPRs, FPIs, saveFROCPath + f"{dataBboxName}_FROC.png", include_FPIs=3)


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization
numSamples=81
visualize(model, dataloader, df, numSamples, saveDir, thres=0.28)