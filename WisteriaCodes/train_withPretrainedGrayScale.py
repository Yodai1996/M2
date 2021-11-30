import sys
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageDraw, ImageFilter
import copy, csv

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import models, transforms, datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, mDice

from network_Gray_VGG import GrayVGG16, GrayVGG16_FC_BN


args = sys.argv

trainPath, validPath, testPath, trainBbox, validBbox, testBbox, modelPath, modelName = args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]
num_epoch, batch_size, numSamples = int(args[9]), int(args[10]), int(args[11])
pretrained, version = args[12], args[13]
trainDir = "/work/gk36/k77012/M2/data/{}/".format(trainPath)
validDir = "/work/gk36/k77012/M2/data/{}/".format(validPath)
testDir = "/work/gk36/k77012/M2/data/{}/".format(testPath)
saveDir = "/work/gk36/k77012/M2/result/{}_{}_{}_batch{}_epoch{}_{}_Dice/".format(trainPath, validPath, modelName, batch_size, num_epoch, pretrained)
df = pd.read_csv("/work/gk36/k77012/M2/{}".format(trainBbox))
df_valid = pd.read_csv("/work/gk36/k77012/M2/{}".format(validBbox))
df_test = pd.read_csv("/work/gk36/k77012/M2/{}".format(testBbox))

# hypara
originalSize = 1024
size = 300
lr = 0.0002
numDisplay = 2 #the number of predicted bboxes to display, also used when calculating mIoU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_df(df, originalSize, size, trainDir)
df_valid = preprocess_df(df_valid, originalSize, size, validDir)
df_test = preprocess_df(df_test, originalSize, size, testDir)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

inDim=3 #as grayScale
trainset = MyDataset(df, transform=transform, inDim=inDim)
validset = MyDataset(df_valid, transform=transform, inDim=inDim)
testset = MyDataset(df_test, transform=transform, inDim=inDim)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1

#use the VGG16 as the backbone pretrained on the gray scale ImageNet.
# grayModel = GrayVGG16_FC_BN().to(device)  # Or use: GrayVGG16_FC_BN().to(device)
print(torch.load("/work/gk36/k77012/M2/GrayVGG16_FC_BN_epoch120_batchsize32.pth"))
# grayModel.load_state_dict(torch.load("/work/gk36/k77012/M2/GrayVGG16_FC_BN_SN_epoch120_batchsize32.pth"))  # grayModel_FC_BN.load_state_dict(torch.load("/work/gk36/k77012/M2/GrayVGG16_FC_BN_SN_epoch120_batchsize32.pth"))
grayModel = torch.load("/work/gk36/k77012/M2/GrayVGG16_FC_BN_epoch120_batchsize32.pth").to(device)
modules = list(grayModel.children())[:4] #7  # for GrayVGG16_FC_BN(), use: modules = list(grayModel.children())[:4]  #as we use the first 4 blocks
backbone = [nn.Conv2d(3,1,1)] #in order to change the num of channel from 3 to 1.
backbone.extend(modules)
sequential = nn.Sequential(*backbone)
if modelName=="SSD":
    model = models.detection.ssd300_vgg16(num_classes = num_classes).to(device) #default: pretrained=False, pretrained_backbone=True
    model.backbone.features = sequential.to(device)  #modify the backbone of the model
    print(model)
# if modelName=="SSD":
#     if pretrained=="pretrained":
#         model = models.detection.ssd300_vgg16(pretrained=True).to(device)
#     else:
#         model = models.detection.ssd300_vgg16(pretrained=False).to(device)
#     model2 = models.detection.ssd300_vgg16(num_classes = num_classes) #models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)
#     model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model
else:  #modelName=="fasterRCNN"
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
    if pretrained == "pretrained":
        model.load_state_dict(torch.load("/work/gk36/k77012/M2/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))  # model.load_state_dict(torch.load("/work/gk36/k77012/faster_RCNN.pth"))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

best_miou = 0
best_mdice = 0
best_epoch = None
best_model = None
test_performance = 0

print(trainset[0][0].shape)

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    # validation
    with torch.no_grad():
        valid_loss = valid(validloader, model)
        test_loss = valid(testloader, model)

        #calculate performance of mean IoU
        miou = mIoU(validloader, model, numDisplay)
        mdice = mDice(validloader, model, numDisplay)
        testmiou = mIoU(testloader, model, numDisplay)
        testmdice= mDice(testloader, model, numDisplay)

        if mdice > best_mdice:
            best_mdice = mdice
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            test_performance = testmdice  # update the test performance
        best_miou = max(best_miou, miou)

    print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mDice:{:.4f}   test_loss:{:.4f}  test_mIoU:{:.4f}  test_mDice:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, mdice, test_loss, testmiou, testmdice))

#print("best_mIoU:{:.4f},   best_mDice:{:.4f}".format(best_miou, best_mdice))
print("best_mDice:{:.4f}   (epoch:{}),   best_mIoU:{:.4f}".format(best_mdice, best_epoch + 1, best_miou))
print("test_mDice:{:.4f}".format(test_performance))

#save the model since we might use it later
PATH = modelPath + "model" + str(version)
torch.save(best_model, PATH) #best_model
model.load_state_dict(torch.load(PATH)) #visualizationのときにもこのbest modelを用いることにする。

#modify and redefine again to use in visualization
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization of training, validation and testing
visualize(model, trainloader, df, numSamples, saveDir + "train/", numDisplay)
visualize(model, validloader, df_valid, numSamples, saveDir + "valid/", numDisplay)
visualize(model, testloader, df_test, numSamples, saveDir + "test/", numDisplay)