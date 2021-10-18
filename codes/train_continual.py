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

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, trainWithCls

args = sys.argv
trainPath, validPath, testPath, trainBbox, validBbox, testBbox, modelName = args[1], args[2], args[3], args[4], args[5], args[6], args[7]
num_epoch = int(args[8])
batch_size = int(args[9])
numSamples = int(args[10])
version = int(args[11])
savePath, modelPath, optimizerName = args[12], args[13], args[14]
if optimizerName=='VSGD':
    variability, decayRate = args[15], args[16]

trainDir, validDir = trainPath, validPath
testDir = "/lustre/gk36/k77012/M2/data/{}/".format(testPath)
saveDir = "/lustre/gk36/k77012/M2/result/{}/".format(savePath)
df = pd.read_csv(trainBbox)
df_valid = pd.read_csv(validBbox)
df_test = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(testBbox))

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
trainset = MyDataset(df, transform=transform)
validset = MyDataset(df_valid, transform=transform)
testset = MyDataset(df_test, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1
pretrained = "pretrained"
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

#load the previously trained model
if version >= 2:
    PATH = modelPath + "model" + str(version-1) #"{}/model{}".format(modelPath, version-1) #"/lustre/gk36/k77012/M2/model/{}/{}/model{}".format(modelPath, optimizerName, version - 1) #version-1 represents the previous step
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
    prevDir = validPath[:barIndexForDir-1] + str(i) + validPath[barIndexForDir:]
    df_prev = pd.read_csv(validBbox[:barIndexForBbox-1] + str(i) + validBbox[barIndexForBbox:])
    df_prev = preprocess_df(df_prev, originalSize, size, prevDir)
    prevset = MyDataset(df_prev, transform=transform)
    prevloader = DataLoader(prevset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
    prevList.append(prevloader)

# training
if optimizerName=='VSGD':
    from optimizers.vsgd import VSGD
    num_iters = len(trainloader) #it will be the ceiling of num_data/batch_size
    variability = variability * (decayRate**version) #by default, variavility=0.01, decayRate=1. #reduce the epsilon by calculating variability * (decayRate)**version
    optimizer = VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters) #VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters, weight_decay=weight_decay)
elif optimizerName=="SAM":
    from optimizers.sam import SAMSGD
    optimizer = SAMSGD(model.parameters(), lr=lr, rho=0.05)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr)


best_miou = 0
best_epoch = None
best_map = 0
test_miou = 0
best_miou_model = None

for epoch in range(num_epoch):

    if optimizerName == "SAM":
        train_loss = trainWithCls(trainloader, model, optimizer)  #SAMの場合はclosureにしないといけなそう.
    else:
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

        # updathe the best performance
        if miou > best_miou:
            best_miou = miou
            best_miou_model = copy.deepcopy(model.state_dict())
            test_miou = testmiou #update the test performance
            best_epoch = epoch

        best_map = max(best_map, map)

    #print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mAP:{:.4f}   test_loss:{:.4f}  test_mIoU:{:.4f}  test_mAP:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, map, test_loss, testmiou, testmap))
    print(
        "epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.3f}  valid_mAP:{:.3f}   test_loss:{:.4f}  test_mIoU:{:.3f}  test_mAP:{:.3f}".format(
            epoch + 1, num_epoch, train_loss, valid_loss, miou, map, test_loss, testmiou, testmap), end="  ")

    #check catastrophic forgetting
    for i,loader in enumerate(prevList):
        sim_i_miou = mIoU(loader, model, numDisplay)
        print("sim{}mIoU:{:.3f}".format(i+1, sim_i_miou), end="  ")

    #改行
    print()

print("best_mIoU:{:.3f}  (epoch:{}),  best_mAP:{:.3f}".format(best_miou, best_epoch+1, best_map))
print("test_mIoU:{:.3f}".format(test_miou))

#save the model since we might use it later
PATH = modelPath + "model" + str(version)  #"{}/model{}".format(modelPath, version) #"/lustre/gk36/k77012/M2/model/{}/model{}".format(modelPath, version)
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
