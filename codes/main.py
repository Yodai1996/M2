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
#from sklearn.metrics import roc_curve, roc_auc_score, recall_score, confusion_matrix
from skimage.transform import resize

from fractalGenerator import make_abnormals
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

from skimage import io, transform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset

'''
Generating abnormal images
'''

args = sys.argv
version, boText, bufText, num_epoch, batch_size, numSamples, scoreThres = int(args[1]), args[2], args[3], int(args[4]), int(args[5]), int(args[6]), float(args[7])

normalDir = "/lustre/gk36/k77012/M2/data/NormalDir1000/0_normal1000/"
segMaskDir = "/lustre/gk36/k77012/M2/data/Mask/mask_{}/".format(version)
trainDir = abnormalDir = "/lustre/gk36/k77012/M2/data/train_{}/1_abnormal1000_{}/".format(version, version)  #現状、異常データだけなので、abnormalDirみたいなもの。
validDir = "/lustre/gk36/k77012/M2/data/AbnormalDir/"
inferDir = "/lustre/gk36/k77012/M2/results/train_{}/".format(version) #saveDir for the visualization

# read the recommended next values from Gaussian Process.
fileHandle = open('/lustre/gk36/k77012/M2/' + bufText, "r")
lineList = fileHandle.readlines()
fileHandle.close()
last_lines = lineList[-1]

if last_lines[-1] == "\n":
    last_lines = last_lines[:-2]

values = last_lines.split(",")
values = [float(i) for i in values]

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

# file name index
with open('/lustre/gk36/k77012/M2/normalIdList1000.csv', 'r') as f:
    read = csv.reader(f)
    ll = list(read)[0]
f.close()

# make abnormal data
print("begin to make abnormal images")
parameterInfo, bboxInfo = make_abnormals(ll, normalDir, abnormalDir, segMaskDir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
print("finish making abnormal images")

# store the information about parameters to reproduce the same images
with open('/lustre/gk36/k77012/M2/simDataInfo/paraInfo/parameterInfo_{}.csv'.format(version), 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(parameterInfo)  # instead of writerow
f.close()

# store the information about bbox to use it in model learning
with open('/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/bboxInfo_{}.csv'.format(version), 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(bboxInfo)  # instead of writerow
f.close()

'''
training a detection model
'''

# hypara
originalSize = 1024
size = 300
lr = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

df = pd.read_csv("/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/bboxInfo_{}.csv".format(version))
df_valid = pd.read_csv("/lustre/gk36/k77012/M2/abnormal_bboxInfo.csv")  #同じ形式に成形した.
df = preprocess_df(df, originalSize, size, trainDir)
df_valid = preprocess_df(df_valid, originalSize, size, validDir)

trainset = MyDataset(df, transform=transform)
validset = MyDataset(df_valid, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")) # model.load_state_dict(torch.load("/lustre/gk36/k77012/faster_RCNN.pth"))

num_classes = 2  # (len(classes)) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

best_loss = float("inf") #initialization
for epoch in range(num_epoch):
    train_loss = train(trainloader, model, optimizer)

    # validation
    with torch.no_grad():
        valid_loss = valid(validloader, model)
        best_loss = min(best_loss, valid_loss)

    print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss))

print("best_loss:{:.4f}".format(best_loss))

# save the values and score
fileHandle = open("/lustre/gk36/k77012/M2/bo_io/in/" + boText, "a")
fileHandle.write("\n" + last_lines + ", " + str(best_loss*(-1)))    ####マイナスをつけてそれを最大化⇔loss自体の最小化、としている事に注意！
fileHandle.close()

# visualization of training
visualize(model, trainset, df, originalSize, size, scoreThres, numSamples, inferDir + "/train/")

# visualization of validation
visualize(model, validset, df_valid, originalSize, size, scoreThres, numSamples, inferDir + "/valid/")


