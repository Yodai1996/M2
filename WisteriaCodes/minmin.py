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
Generating abnormal images
'''

args = sys.argv
version, boText, bufText, normalIdList, normalDir, abnormalDir, segMaskDir, saveParaPath, saveBboxPath = int(args[1]), args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]
validPath, testPath, savePath, validBboxName, testBboxName, modelName, pretrained = args[10], args[11], args[12], args[13], args[14], args[15], args[16]
num_epoch, batch_size, numSamples = int(args[17]), int(args[18]), int(args[19])
modelPath, metric, optimizerName = args[20], args[21], args[22]

validBbox, testBbox = validBboxName + ".csv", testBboxName + ".csv"  #valid と testだけBboxNameで引数渡した.

trainDir = abnormalDir
validDir = "/work/gk36/k77012/M2/data/{}/".format(validPath)
testDir = "/work/gk36/k77012/M2/data/{}/".format(testPath)
saveDir = "/work/jh170036a/k77012/M2/result/{}/".format(savePath)

# read the recommended next values from Gaussian Process.
fileHandle = open('/work/gk36/k77012/M2/' + bufText, "r")
lineList = fileHandle.readlines()
fileHandle.close()
last_lines = lineList[-1]

if last_lines[-1] == "\n":
    last_lines = last_lines[:-2]

values = last_lines.split(",")
values = [float(i) for i in values]

# preprocess
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

# file name index
with open(normalIdList, 'r') as f:
    read = csv.reader(f)
    ll = list(read)
f.close()

# 一応両方バージョン書いとく。
if len(ll) <= 2:  # １行版
    ll = ll[0]  # [ll[0][i] for i in range(len(ll))]
else:  # 改行版
    ll = [ll[i][0] for i in range(len(ll))]

# make abnormal data
print("begin to make abnormal images")
parameterInfo, bboxInfo = make_abnormals(ll, normalDir, abnormalDir, segMaskDir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
print("finish making abnormal images")

# store the information about parameters to reproduce the same images
with open(saveParaPath, 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(parameterInfo)  # instead of writerow

# store the information about bbox to use it in model learning
with open(saveBboxPath, 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(bboxInfo)  # instead of writerow


'''
training a detection model
'''

# hypara
originalSize = 1024
size = 300
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

df = pd.read_csv(saveBboxPath)
df_valid = pd.read_csv("/work/gk36/k77012/M2/{}".format(validBbox))
df_test = pd.read_csv("/work/gk36/k77012/M2/{}".format(testBbox))

df = preprocess_df(df, originalSize, size, trainDir)
df_valid = preprocess_df(df_valid, originalSize, size, validDir)
df_test = preprocess_df(df_test, originalSize, size, testDir)

trainset = MyDataset(df, transform=transform)
validset = MyDataset(df_valid, transform=transform)
testset = MyDataset(df_test, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
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
        loadModelPath="/work/gk36/k77012/M2/model/pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120" #とりあえず、VSGD, noise=0.01を使用すれことにする。
        model.load_state_dict(torch.load(loadModelPath))

# training
if optimizerName=='VSGD':
    from optimizers.vsgd import VSGD
    num_iters = len(trainloader) #it will be the ceiling of num_data/batch_size
    variability = 0.01 #fixed for master thesis
    optimizer = VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters) #VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters, weight_decay=weight_decay)
else:
    if optimizerName == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

best_value = 0
best_epoch = -100 #as default
best_model = None
test_performance = 0
test_rcpm = 0

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    #thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]
    # thresholds = [0.01 * i for i in range(101)]
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]

    #see the performance on the training dataset
    TPRs, FPIs, thresholds = FROC(trainloader, model, thresholds=thresholds) #, ignore_big_bbox=Trueは合ってもなくても同じ。そもそも大bboxはないので。
    fauc_train = FAUC(TPRs, FPIs)
    rcpm_train = RCPM(TPRs, FPIs)

    # validation
    with torch.no_grad():

        # #不要だがまあ一応。
        # TPRs, FPIs, thresholds = FROC(validloader, model, thresholds=thresholds)
        # fauc = FAUC(TPRs, FPIs)
        # rcpm = RCPM(TPRs, FPIs)

        #Ignore Big
        TPRs, FPIs, thresholds = FROC(validloader, model, thresholds=thresholds, ignore_big_bbox=True)
        fauc_IB = FAUC(TPRs, FPIs)
        rcpm_IB = RCPM(TPRs, FPIs)

        if fauc_IB > best_value:
            best_value = fauc_IB
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            TPRs, FPIs, thresholds = FROC(testloader, model, thresholds=thresholds, ignore_big_bbox=True)
            test_performance = FAUC(TPRs, FPIs)
            test_rcpm = RCPM(TPRs, FPIs)

    #print("epoch:{}/{}  tr_loss:{:.4f}   tr_fauc:{:.4f}   tr_rcpm:{:.4f}    val_fauc:{:.4f}   val_rcpm:{:.4f}   val_fauc_IB:{:.4f}  val_rcpm_IB:{:.4f}".format(epoch + 1, num_epoch, train_loss, fauc_train, rcpm_train, fauc, rcpm, fauc_IB, rcpm_IB)) #strict is deleted
    print("epoch:{}/{}  tr_loss:{:.4f}   tr_fauc:{:.4f}   tr_rcpm:{:.4f}    val_fauc_IB:{:.4f}  val_rcpm_IB:{:.4f}".format(epoch + 1, num_epoch, train_loss, fauc_train, rcpm_train, fauc_IB, rcpm_IB)) #strict is deleted

print("best_fauc_IgnoreBigBbox:{:.4f}   (epoch:{})".format(best_value, best_epoch + 1))
print("test_fauc_IgnoreBigBbox:{:.4f},  test_rcpm_IgnoreBigBbox:{:.4f}    At the epoch.".format(test_performance, test_rcpm))


# save the values and score for the next iteration
fileHandle = open("/work/gk36/k77012/M2/bo_io/in/" + boText, "a")
obj_function = best_value #metric is assumed as FAUC
fileHandle.write("\n" + last_lines + ", " + str(obj_function))
fileHandle.close()


#save the model since we might use it later
PATH = modelPath + f"model_version{version}_{validBboxName}_pretrained{pretrained}_epoch{num_epoch}"
torch.save(best_model, PATH) #best_model
model.load_state_dict(torch.load(PATH)) #visualizationのときにもこのbest modelを用いることにする。

#modify and redefine again to use in visualization
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization of training, validation and testing
visualize(model, trainloader, df, numSamples, saveDir + "train/", thres=0.3)
visualize(model, validloader, df_valid, numSamples, saveDir + "valid/", thres=0.3)
visualize(model, testloader, df_test, numSamples, saveDir + "test/", thres=0.3)
