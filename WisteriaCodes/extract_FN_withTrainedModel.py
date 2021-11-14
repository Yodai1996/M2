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

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, returnFnIndices, returnFnIndices_Dice

args = sys.argv
ver, dataPath, dataBbox, thres, lower_thres, saveDir, saveFnDir = int(args[1]), args[2], args[3], float(args[4]), float(args[5]), args[6], args[7]
#lower_thres = -1 #すでに得られたFN_indicesとの重複を消すため。 #引数でもいいから、そうした。
dataDir = "/work/gk36/k77012/M2/data/{}/".format(dataPath)
df = pd.read_csv("/work/gk36/k77012/M2/{}".format(dataBbox))

# hypara
originalSize = 1024
size = 300
batch_size = 64 #any value might be fine.
numDisplay = 2 #the number of predicted bboxes to display, also used when calculating mIoU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_df(df, originalSize, size, dataDir)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])
dataset = MyDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn) #shuffle must be False!!

num_classes = 2 #(len(classes)) + 1

#only consider SSD model
model = models.detection.ssd300_vgg16(pretrained=False, pretrained_backbone=False).to(device)
model2 = models.detection.ssd300_vgg16(num_classes=num_classes)
model.head.classification_head = model2.head.classification_head.to(device)  # modify the head of the model

#load the saved, trained model
PATH = "/work/gk36/k77012/M2/model/model_FN_search_{}".format(ver)
model.load_state_dict(torch.load(PATH)) #visualizationのときにもこのbest modelを用いることにする。

#search FN indices and save
with torch.no_grad():
    FN_indices = returnFnIndices_Dice(dataloader, model, numDisplay, thres=thres, lower_thres=lower_thres) #used deice instead of iou

with open('/work/gk36/k77012/M2/FN_indices_among1176_{}_thres{}_lowerThres{}.csv'.format(ver, thres, lower_thres), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(FN_indices)
f.close()

#re-define
#以下は、１度やれば十分。
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn) #shuffle must be False!!
#FN_indicesだけでよいため、無駄が発生するが、一応全部保存しておく。
numSamples = 1176
visualize(model, dataloader, df, numSamples, saveDir + "Images_with_gt_and_inference/", numDisplay)
visualize(model, dataloader, df, numSamples, saveDir + "Images_with_gt/", numDisplay=0) #gt_bboxだけにする。


'''
below part is to extract and save FN data
'''
FN_files = []
for ind in FN_indices:
    ind = int(ind)
    FN_files.append(df["file"][ind])

#save data
for file in FN_files:
    img = Image.open(saveDir + "Images_with_gt_and_inference/" + file)
    img.save(saveFnDir + "Images_with_gt_and_inference/" + file)

    img = Image.open(saveDir + "Images_with_gt/" + file)
    img.save(saveFnDir + "Images_with_gt/" + file)

    img = Image.open(dataDir + file)
    img.save(saveFnDir + "/Original_images/" + file)

#save the file names
with open('/work/gk36/k77012/M2/FN_files_among1176_{}_thres{}_lowerThres{}.csv'.format(ver, thres, lower_thres), 'w', newline='') as f:
    writer = csv.writer(f)
    for v in FN_files:
        writer.writerow([v])
f.close()
