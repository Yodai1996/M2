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

args = sys.argv
dataPath, dataBbox, numSamples, maxsize, saveDir = args[1], args[2], int(args[3]), int(args[4]), args[5]
dataDir = "/work/gk36/k77012/M2/data/{}/".format(dataPath)
df = pd.read_csv("/work/gk36/k77012/M2/{}".format(dataBbox))

# hypara
originalSize = 1024
size = 300
batch_size = 64 #any value might be fine.
numDisplay = 0 #the number of predicted bboxes to display, also used when calculating mIoU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_df(df, originalSize, size, dataDir)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])
dataset = MyDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn) #shuffle must be False!!

modelName="SSD"
pretrained="pretrained"
num_classes = 2 #(len(classes)) + 1
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

# visualization of training, validation and testing
visualize(model, dataloader, df, numSamples, saveDir + "Original_images/", numDisplay=0, show_gt=False, show_inf=False)
visualize(model, dataloader, df, numSamples, saveDir + "Images_with_gt/", numDisplay=0, maxsize=maxsize*size/originalSize) #画素数がさがっているため、調整が必要.
