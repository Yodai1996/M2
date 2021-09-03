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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, miou

args = sys.argv
trainPath, validPath, testPath, trainBbox, validBbox, testBbox, modelName = args[1], args[2], args[3], args[4], args[5], args[6], args[7]
num_epoch = int(args[8])
batch_size = int(args[9])
numSamples = int(args[10])
trainDir = "/lustre/gk36/k77012/M2/data/{}/".format(trainPath)
validDir = "/lustre/gk36/k77012/M2/data/{}/".format(validPath)
testDir = "/lustre/gk36/k77012/M2/data/{}/".format(testPath)
#inferDir = "/lustre/gk36/k77012/M2/results/{}_{}_{}_batch{}_epoch{}/".format(trainPath, validPath, modelName, batch_size, num_epoch)
inferDir = "/lustre/gk36/k77012/M2/result/{}_{}_{}_batch{}_epoch{}/".format(trainPath, validPath, modelName, batch_size, num_epoch)
df = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(trainBbox))
df_valid = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(validBbox))
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

#if modelName=="SSD":
model = models.detection.ssd300_vgg16(pretrained=False, pretrained_backbone=False).to(device)
model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/ssd300_vgg16_coco-b556d3b4.pth"))

num_classes = 2 #(len(classes)) + 1
model2 = models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)

#modify the head of the model
model.head.classification_head = model2.head.classification_head.to(device)


# training
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    # validation
    with torch.no_grad():
        valid_loss = valid(validloader, model)
        test_loss = valid(testloader, model)

        #calculate performance of mean IoU
        mIoU = miou(validloader, model, numDisplay)
        testmIoU = miou(testloader, model, numDisplay)

    #print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  test_loss:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, test_loss))
    print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  test_loss:{:.4f}  test_mIoU:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, mIoU, test_loss, testmIoU))

#modify and redefine again to use in visualization
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization of training, validation and testing
visualize(model, trainloader, df, numSamples, inferDir + "train/", numDisplay)
visualize(model, validloader, df_valid, numSamples, inferDir + "valid/", numDisplay)
visualize(model, testloader, df_test, numSamples, inferDir + "test/", numDisplay)
