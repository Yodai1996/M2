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

args = sys.argv
trainPath, validPath, realValidPath, testPath, trainBbox, validBbox, realValidBbox, testBbox, modelPath, modelName = args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]
num_epoch, batch_size, numSamples, version = int(args[11]), int(args[12]), int(args[13]), int(args[14])
savePath, optimizerName, pretrained = args[15], args[16], args[17]
if optimizerName=='VSGD':
    variability, decayRate = float(args[18]), float(args[19])

trainDir, validDir = trainPath, validPath
realValidDir = "/work/gk36/k77012/M2/data/{}/".format(realValidPath)
testDir = "/work/gk36/k77012/M2/data/{}/".format(testPath)
saveDir = "/work/gk36/k77012/M2/result/{}/".format(savePath)
df = pd.read_csv(trainBbox)
df_valid = pd.read_csv(validBbox)
df_realValid = pd.read_csv("/work/gk36/k77012/M2/{}".format(realValidBbox))
df_test = pd.read_csv("/work/gk36/k77012/M2/{}".format(testBbox))

# hypara
originalSize = 1024
size = 300
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_df(df, originalSize, size, trainDir)
df_valid = preprocess_df(df_valid, originalSize, size, validDir)
df_realValid = preprocess_df(df_realValid, originalSize, size, realValidDir)
df_test = preprocess_df(df_test, originalSize, size, testDir)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])
trainset = MyDataset(df, transform=transform)
validset = MyDataset(df_valid, transform=transform)
realValidset = MyDataset(df_realValid, transform=transform)
testset = MyDataset(df_test, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
realValidloader = DataLoader(realValidset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1
if modelName=="SSD":
    if pretrained=="pretrained" or pretrained=="ImageNet": #same meaning
        model = models.detection.ssd300_vgg16(pretrained=True).to(device)
    else:
        model = models.detection.ssd300_vgg16(pretrained=False).to(device)
    model2 = models.detection.ssd300_vgg16(num_classes = num_classes) #models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)
    model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model

    if pretrained=="BigBbox":
        loadModelPath=modelPath+"pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120" #とりあえず、VSGD, noise=0.01を使用すれことにする。
        model.load_state_dict(torch.load(loadModelPath))

#load the previously trained model
if version >= 2:
    PATH = modelPath + "model" + str(version-1)
    model.load_state_dict(torch.load(PATH))

#will be used to check catastrophic forgetting
prevList = []
barIndexForDir, barIndexForBbox = None, None
#we have to use the last _ exisiting in the string, as follows.
for i, v in enumerate(validPath):
    if v=="_": #find the index representing the bar
        barIndexForDir = i
for i, v in enumerate(validBbox):
    if v=="_": #find the index representing the bar
        barIndexForBbox = i

for i in range(1,version):  #[1,version)
    prevDir = validPath[:barIndexForDir-len(str(version))] + str(i) + validPath[barIndexForDir:]
    df_prev = pd.read_csv(validBbox[:barIndexForBbox-len(str(version))] + str(i) + validBbox[barIndexForBbox:])
    df_prev = preprocess_df(df_prev, originalSize, size, prevDir)
    prevset = MyDataset(df_prev, transform=transform)
    prevloader = DataLoader(prevset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
    prevList.append(prevloader)

# training
if optimizerName=='VSGD':
    from optimizers.vsgd import VSGD
    num_iters = len(trainloader) #it will be the ceiling of num_data/batch_size
    variability = variability * (decayRate**version) #by default, variability=0.01, decayRate=1. #reduce the epsilon by calculating variability * (decayRate)**version
    optimizer = VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters) #VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters, weight_decay=weight_decay)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr)

best_value = 0
best_epoch = -100 #as default
best_model = None
realValid_performance = 0
test_performance = 0

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    #see the performance on the training dataset
    TPRs, FPIs, thresholds = FROC(trainloader, model) #, ignore_big_bbox=Trueは合ってもなくても同じ。そもそも大bboxはないので。
    fauc_train = FAUC(TPRs, FPIs)
    rcpm_train = RCPM(TPRs, FPIs)
    #cpm_train  = CPM(TPRs, FPIs)

    # validation
    with torch.no_grad():

        #Ignore Big
        TPRs, FPIs, thresholds = FROC(validloader, model, ignore_big_bbox=True)
        fauc_IB = FAUC(TPRs, FPIs)
        rcpm_IB = RCPM(TPRs, FPIs)
        cpm_IB  = CPM(TPRs, FPIs)

        #Ignore Big
        TPRs, FPIs, thresholds = FROC(realValidloader, model, ignore_big_bbox=True)
        fauc_realValid_IB = FAUC(TPRs, FPIs)
        rcpm_realValid_IB = RCPM(TPRs, FPIs)
        cpm_realValid_IB  = CPM(TPRs, FPIs)

        if fauc_IB > best_value:
            best_value = fauc_IB
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            realValid_performance = fauc_realValid_IB

            TPRs, FPIs, thresholds = FROC(testloader, model, thresholds=thresholds, ignore_big_bbox=True)
            test_performance = FAUC(TPRs, FPIs)

    print("epoch:{}/{}  tr_loss:{:.4f}   tr_fauc:{:.4f}   tr_rcpm:{:.4f}   val_fauc:{:.4f}  val_cpm:{:.4f}  val_rcpm:{:.4f}   realVal_fauc:{:.4f}   realVal_cpm:{:.4f}  realVal_rcpm:{:.4f}".format(epoch + 1, num_epoch, train_loss, fauc_train, rcpm_train, fauc_IB, cpm_IB, rcpm_IB, fauc_realValid_IB, cpm_realValid_IB, rcpm_realValid_IB)) #strict is deleted

    #check catastrophic forgetting
    for i,loader in enumerate(prevList):
        TPRs, FPIs, thresholds = FROC(loader, model, ignore_big_bbox=True)
        sim_i_fauc = FAUC(TPRs, FPIs)
        print("sim{}FAUC:{:.3f}".format(i+1, sim_i_fauc), end="  ")

    #改行
    print()

print("best_fauc_IgnoreBigBbox:{:.4f}   (epoch:{})".format(best_value, best_epoch + 1))
print("---At the epoch, performances are as follows:---")
print("realValid_FAUC:{:.3f}".format(realValid_performance))
print("test_FAUC:{:.3f}".format(test_performance))

#save the model since we might use it later
PATH = modelPath + "model" + str(version)
torch.save(best_model, PATH) #best_model
model.load_state_dict(torch.load(PATH)) #visualizationのときにもこのbest modelを用いることにする。

#modify and redefine again to use in visualization
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
#realValidloader = DataLoader(realValidset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn) #不要そうではあるので。
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization of training, validation and testing
visualize(model, trainloader, df, numSamples, saveDir + "train/", thres=0.3)
visualize(model, validloader, df_valid, numSamples, saveDir + "valid/", thres=0.3)
visualize(model, realValidloader, df_realValid, numSamples, saveDir + "realValid/", thres=0.3)
visualize(model, testloader, df_test, numSamples, saveDir + "test/", thres=0.3)