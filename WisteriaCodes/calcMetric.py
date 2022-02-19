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
To see the performances in the following setting, normal:abnormal=10:1
'''

args = sys.argv
dataPath, validBboxName, testBboxName, Path, loadModelPath, pretrained = args[1], args[2], args[3], args[4], args[5], args[6]

validBbox = validBboxName + ".csv"
testBbox = testBboxName + ".csv"
dataDir = "/work/gk36/k77012/M2/data/{}/".format(dataPath)
df_valid = pd.read_csv("/work/gk36/k77012/M2/{}".format(validBbox))
df_test = pd.read_csv("/work/gk36/k77012/M2/{}".format(testBbox))

# hypara
originalSize = 1024
size = 300
batch_size=64 #ただのinferenceなので、batch sizeは何でもよい
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

df_valid = preprocess_df(df_valid, originalSize, size, dataDir)
df_test = preprocess_df(df_test, originalSize, size, dataDir)
validset = MyDataset(df_valid, transform=transform)
testset = MyDataset(df_test, transform=transform)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

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
TPRs, FPIs, thresholds = FROC(validloader, model, ignore_big_bbox=True)
fauc = FAUC(TPRs, FPIs)
cpm = CPM(TPRs, FPIs)
rcpm = RCPM(TPRs, FPIs)
print("valid data, i.e., #normal=200, #abnormal=20")
print("valid_fauc:{:.4f}  valid_cpm:{:.4f}  valid_rcpm:{:.4f}".format(fauc, cpm, rcpm))

print()

TPRs, FPIs, thresholds = FROC(testloader, model, ignore_big_bbox=True)
fauc = FAUC(TPRs, FPIs)
cpm = CPM(TPRs, FPIs)
rcpm = RCPM(TPRs, FPIs)
print("test data, i.e., #normal=810, #abnormal=81")
print("test_fauc:{:.4f}  test_cpm:{:.4f}  test_rcpm:{:.4f}".format(fauc, cpm, rcpm))

#ついでにBigBboxでの破滅忘却を見るためにパフォーマンスチェックする

if pretrained=="BigBbox":
    df_bigbbox = pd.read_csv("/work/gk36/k77012/M2/nonSmall_bboxInfo_164_withNormal.csv")
    df_bigbbox = preprocess_df(df_bigbbox, originalSize, size, dataDir) #just AllDataDir
    pretrainset = MyDataset(df_bigbbox, transform=transform)
    pretrainloader = DataLoader(pretrainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

    TPRs, FPIs, thresholds = FROC(pretrainloader, model, thresholds=thresholds) #ignore_big_bbox=False, accept_TP_duplicate=True
    fauc = FAUC(TPRs, FPIs)
    cpm = CPM(TPRs, FPIs)
    rcpm = RCPM(TPRs, FPIs)
    print()
    print("The performances on the initial task, for BigBbox-pretrained model")
    print("BigBbox_fauc:{:.4f}  BigBbox_cpm:{:.4f}  BigBbox_rcpm:{:.4f}".format(fauc, cpm, rcpm))