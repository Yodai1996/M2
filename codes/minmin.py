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
from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP

'''
Generating abnormal images
'''

args = sys.argv
version, boText, bufText, normalIdList, normalDir, abnormalDir, segMaskDir, saveParaPath, saveBboxPath = int(args[1]), args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]
validPath, testPath, savePath, validBbox, testBbox, modelName, pretrained = args[10], args[11], args[12], args[13], args[14], args[15], args[16]
num_epoch, batch_size, numSamples = int(args[17]), int(args[18]), int(args[19])

trainDir = abnormalDir
validDir = "/lustre/gk36/k77012/M2/data/{}/".format(validPath)
testDir = "/lustre/gk36/k77012/M2/data/{}/".format(testPath)
saveDir = "/lustre/gk36/k77012/M2/result/{}/".format(savePath)

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
with open(normalIdList, 'r') as f:
    read = csv.reader(f)
    ll = list(read)[0]
f.close()

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
numDisplay = 2 #the number of predicted bboxes to display, also used when calculating mIoU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

df = pd.read_csv(saveBboxPath)
df_valid = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(validBbox))
df_test = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(testBbox))

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

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

best_miou, best_map = 0, 0
best_miou_model = None

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    # validation
    with torch.no_grad():
        valid_loss = valid(validloader, model)
        test_loss = valid(testloader, model)

        #calculate performance of mean IoU
        miou = mIoU(validloader, model, numDisplay)
        map = mAP(validloader, model, numDisplay)
        testmiou = mIoU(testloader, model, numDisplay)
        testmap = mAP(testloader, model, numDisplay)

        #updathe the best performance
        if miou > best_miou:
            best_miou = miou
            best_miou_model = copy.deepcopy(model.state_dict())

        best_map = max(best_map, map)

    print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mAP:{:.4f}   test_loss:{:.4f}  test_mIoU:{:.4f}  test_mAP:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, map, test_loss, testmiou, testmap))
    #print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mAP:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, map))

print("best_mIoU:{:.4f},   best_mAP:{:.4f}".format(best_miou, best_map))


#save the model since we might use it later
PATH = "/lustre/gk36/k77012/M2/model/minmin/model{}".format(version)
torch.save(best_miou_model, PATH) #best_miou_model
model.load_state_dict(torch.load(PATH)) #visualizationのときにもこのbest modelを用いることにする。

#modify and redefine again to use in visualization
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization of training, validation and testing
visualize(model, trainloader, df, numSamples, saveDir + "train/", numDisplay)
visualize(model, validloader, df_valid, numSamples, saveDir + "valid/", numDisplay)
visualize(model, testloader, df_test, numSamples, saveDir + "test/", numDisplay)

# save the values and score for the next iteration
fileHandle = open("/lustre/gk36/k77012/M2/bo_io/in/" + boText, "a")
fileHandle.write("\n" + last_lines + ", " + str(best_miou))
fileHandle.close()
