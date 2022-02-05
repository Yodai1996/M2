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
from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, mDice, FROC, FAUC, CPM, RCPM, plotFROC, decideThres, calcMetric

'''
just for visualizing
'''

args = sys.argv
dataPath, validBboxName, testBboxName, loadModelPath, saveDir, FPsI = args[1], args[2], args[3], args[4], args[5], float(args[6])

dataDir = "/work/gk36/k77012/M2/data/{}/".format(dataPath)
validBbox = validBboxName + ".csv"
testBbox = testBboxName + ".csv"
df_valid = pd.read_csv("/work/gk36/k77012/M2/{}".format(validBbox))
df_test = pd.read_csv("/work/gk36/k77012/M2/{}".format(testBbox))

# hypara
originalSize = 1024
maxsize = 150 #small bbox size
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
    model = models.detection.ssd300_vgg16(pretrained=False).to(device)
    model2 = models.detection.ssd300_vgg16(num_classes = num_classes)
    model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model
    model.load_state_dict(torch.load(loadModelPath))

#calculate threshold that satisfy under the FPsI, using the validation data
t, tpr, fps, index = decideThres(validloader, model, FPsI)
print("thres for FPsI={} is {}".format(FPsI, t))
print("At the threshold:")
print("validation: TPR={}, FPsI={}".format(tpr, fps))
tpr, fps = calcMetric(testloader, model, index)
print("test: TPR={}, FPsI={}".format(tpr, fps))


# visualization
numSamples=20*2
visualize(model, validloader, df_valid, numSamples, saveDir + validBboxName + "/", thres=t, maxsize=maxsize*size/originalSize)

numSamples=81*2
visualize(model, testloader, df_test, numSamples, saveDir + testBboxName + "/", thres=t, maxsize=maxsize*size/originalSize)