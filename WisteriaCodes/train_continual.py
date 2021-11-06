import sys
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageDraw, ImageFilter

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import models, transforms, datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, trainWithCls, mDice

args = sys.argv
trainPath, validPath, realValidPath, testPath, trainBbox, validBbox, realValidBbox, testBbox, modelPath, modelName = args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]
num_epoch, batch_size, numSamples, version = int(args[11]), int(args[12]), int(args[13]), int(args[14])
savePath, optimizerName, metric = args[15], args[16], args[17]
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
numDisplay = 2 #the number of predicted bboxes to display, also used when calculating mIoU.
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
pretrained = "pretrained"
if modelName=="SSD":
    if pretrained=="pretrained":
        model = models.detection.ssd300_vgg16(pretrained=True).to(device)
    else:
        model = models.detection.ssd300_vgg16(pretrained=False).to(device)
    model2 = models.detection.ssd300_vgg16(num_classes = num_classes) #models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)
    model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model
else:  #modelName=="fasterRCNN"
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
    if pretrained == "pretrained":
        model.load_state_dict(torch.load("/work/gk36/k77012/M2/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))  # model.load_state_dict(torch.load("/work/gk36/k77012/faster_RCNN.pth"))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

#load the previously trained model
if version >= 2:
    PATH = modelPath + "model" + str(version-1) #"{}/model{}".format(modelPath, version-1) #"/work/gk36/k77012/M2/model/{}/{}/model{}".format(modelPath, optimizerName, version - 1) #version-1 represents the previous step
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
elif optimizerName=="SAM":
    from optimizers.sam import SAMSGD
    optimizer = SAMSGD(model.parameters(), lr=lr, rho=0.05)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr)


best_miou = 0
best_epoch = None
best_mdice = 0
#best_map = 0
best_model = None
realValid_performance = 0
test_performance = 0

for epoch in range(num_epoch):

    if optimizerName == "SAM":
        train_loss = trainWithCls(trainloader, model, optimizer)  #SAMの場合はclosureにしないといけなそう.
    else:
        train_loss = train(trainloader, model, optimizer)

    # validation
    with torch.no_grad():
        valid_loss = valid(validloader, model)
        realValid_loss = valid(realValidloader, model)
        #test_loss = valid(testloader, model)

        #calculate performance of mean IoU
        miou = mIoU(validloader, model, numDisplay)
        mdice = mDice(validloader, model, numDisplay)
        realValmiou = mIoU(realValidloader, model, numDisplay)
        realValmdice= mDice(realValidloader, model, numDisplay)
        testmiou = mIoU(testloader, model, numDisplay)
        testmdice= mDice(testloader, model, numDisplay)

        # updathe the best performance
        if metric=="IoU":
            #update IoU
            if miou > best_miou:
                best_miou = miou
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                realValid_performance = realValmiou
                test_performance = testmiou #update the test performance
            best_mdice = max(best_mdice, mdice)
        else: #metric=="Dice"
            #update dice
            if mdice > best_mdice:
                best_mdice = mdice
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                realValid_performance = realValmdice
                test_performance = testmdice #update the test performance
            best_miou = max(best_miou, miou)

        #best_map = max(best_map, map)


    print(
        "epoch:{}/{}  tr_loss:{:.3f}  val_loss:{:.3f}  val_IoU:{:.3f}  val_Dice:{:.3f}  realVal_loss:{:.3f}  realVal_IoU:{:.3f}  realVal_Dice:{:.3f}  test_IoU:{:.3f}  test_Dice:{:.3f}".format(
            epoch + 1, num_epoch, train_loss, valid_loss, miou, mdice, realValid_loss, realValmiou, realValmdice, testmiou, testmdice), end="  ")

    #check catastrophic forgetting
    if metric == "IoU":
        for i,loader in enumerate(prevList):
            sim_i_miou = mIoU(loader, model, numDisplay)
            print("sim{}IoU:{:.3f}".format(i+1, sim_i_miou), end="  ")
    else:  # metric=="Dice"
        for i,loader in enumerate(prevList):
            sim_i_mdice = mDice(loader, model, numDisplay)
            print("sim{}Dice:{:.3f}".format(i+1, sim_i_mdice), end="  ")

    #改行
    print()

#printing the results
if metric == "IoU":
    print("best_mIoU:{:.3f}  (epoch:{}),  best_mDice:{:.3f}".format(best_miou, best_epoch + 1, best_mdice))
    print("---At the epoch, performances are as follows:---")
    print("realValid_mIoU:{:.3f}".format(realValid_performance))
    print("test_mIoU:{:.3f}".format(test_performance))
else: # metric=="Dice"
    print("best_mDice:{:.3f}  (epoch:{}),  best_mIoU:{:.3f}".format(best_mdice, best_epoch + 1, best_miou))
    print("---At the epoch, performances are as follows:---")
    print("realValid_mDice:{:.3f}".format(realValid_performance))
    print("test_mDice:{:.3f}".format(test_performance))

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
visualize(model, trainloader, df, numSamples, saveDir + "train/", numDisplay)
visualize(model, validloader, df_valid, numSamples, saveDir + "valid/", numDisplay)
visualize(model, realValidloader, df_realValid, numSamples, saveDir + "realValid/", numDisplay)
visualize(model, testloader, df_test, numSamples, saveDir + "test/", numDisplay)
