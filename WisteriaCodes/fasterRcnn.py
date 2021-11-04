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

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset

args = sys.argv
trainPath, validPath, testPath, trainBbox, validBbox, testBbox = args[1], args[2], args[3], args[4], args[5], args[6]
num_epoch = int(args[7])
batch_size = int(args[8])
numSamples = int(args[9])
trainDir = "/lustre/gk36/k77012/M2/data/{}/".format(trainPath)
validDir = "/lustre/gk36/k77012/M2/data/{}/".format(validPath)
testDir = "/lustre/gk36/k77012/M2/data/{}/".format(testPath)
inferDir = "/lustre/gk36/k77012/M2/results/{}_{}_batch{}_epoch{}/".format(trainPath, validPath, batch_size, num_epoch)
df = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(trainBbox))
df_valid = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(validBbox))
df_test = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(testBbox))

# hypara
originalSize = 1024
size = 300
lr = 0.0002
scoreThres = 0.3  # used in inference, you can set as you like.
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

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(device)
model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")) # model.load_state_dict(torch.load("/lustre/gk36/k77012/faster_RCNN.pth"))

num_classes = 2  # (len(classes)) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    # validation
    with torch.no_grad():
        valid_loss = valid(validloader, model)
        test_loss = valid(testloader, model)

    print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  test_loss:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, test_loss))

#modify and redefine again to use in visualization
#use batch_size=numSamples
# trainloader = DataLoader(trainset, batch_size=numSamples, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
# validloader = DataLoader(validset, batch_size=numSamples, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization of training
#visualize(model, trainset, df, originalSize, size, numSamples, inferDir + "/train/", numDisplay=2)
visualize(model, trainloader, df, numSamples, inferDir + "train/", numDisplay=2)

# visualization of validation
#visualize(model, validset, df_valid, originalSize, size, numSamples, inferDir + "/valid/", numDisplay=2)
visualize(model, validloader, df_valid, numSamples, inferDir + "valid/", numDisplay=2)

# visualization of test
visualize(model, testloader, df_test, numSamples, inferDir + "test/", numDisplay=2)


# scoreThresではなく上位２つで表示させることにした
# # visualization of training
# visualize(model, trainset, df, originalSize, size, scoreThres, numSamples, inferDir + "/train/")
#
# # visualization of validation
# visualize(model, validset, df_valid, originalSize, size, scoreThres, numSamples, inferDir + "/valid/")
