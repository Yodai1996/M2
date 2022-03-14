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

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, mDice, FROC, FAUC, CPM, RCPM, plotFROC

args = sys.argv

trainPath, validPath, trainBboxName, validBboxName, loadModelPath, saveModelPath = args[1], args[2], args[3], args[4], args[5], args[6]
modelName, num_epoch, batch_size, = args[7], int(args[8]), int(args[9])
optimizerName, variability = args[10], float(args[11])

trainBbox, validBbox = trainBboxName + ".csv", validBboxName + ".csv"
trainDir = "/work/gk36/k77012/M2/data/{}/".format(trainPath)
validDir = "/work/gk36/k77012/M2/data/{}/".format(validPath)
df = pd.read_csv("/work/gk36/k77012/M2/{}".format(trainBbox))
df_valid = pd.read_csv("/work/gk36/k77012/M2/{}".format(validBbox))

# hypara
originalSize = 1024
size = 300
lr = 0.0002
#isplay = 2 #the number of predicted bboxes to display, also used when calculating mIoU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_df(df, originalSize, size, trainDir)
df_valid = preprocess_df(df_valid, originalSize, size, validDir)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

inDim = 3 #3
trainset = MyDataset(df, transform=transform, inDim=inDim)
validset = MyDataset(df_valid, transform=transform, inDim=inDim)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1
if modelName=="SSD":
    model = models.detection.ssd300_vgg16(pretrained=False).to(device)
    model2 = models.detection.ssd300_vgg16(num_classes = num_classes) #models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)
    model.head.classification_head = model2.head.classification_head.to(device)  #modify the head of the model
    model.load_state_dict(torch.load(loadModelPath))


# training
if optimizerName=='VSGD':
    from optimizers.vsgd import VSGD
    num_iters = len(trainloader) #it will be the ceiling of num_data/batch_size
    optimizer = VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters) #VSGD(model.parameters(), lr=lr, variability=variability, num_iters=num_iters, weight_decay=weight_decay)
elif optimizerName == "SAM":
    from optimizers.sam import SAMSGD
    optimizer = SAMSGD(model.parameters(), lr=lr, rho=variability)
else:
    if optimizerName=="Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)


best_value = 0
best_epoch = None
best_model = None

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    # validation
    with torch.no_grad():
        #valid_loss = valid(validloader, model)

        #calculate performance of mean IoU
        #dice = mDice(validloader, model)
        TPRs, FPIs, thresholds = FROC(validloader, model)
        # print(TPRs)
        # print(FPIs)
        # print(thresholds)
        fauc = FAUC(TPRs, FPIs) #FAUC(TPRs, FPIs, thresholds)
        cpm = CPM(TPRs, FPIs)
        rcpm = RCPM(TPRs, FPIs)

        ###visualize here
        TPRs, FPIs, thresholds = FROC(validloader, model, accept_TP_duplicate=False)
        # print(TPRs)
        # print(FPIs)
        # print(thresholds)
        fauc_strict = FAUC(TPRs, FPIs) #FAUC(TPRs, FPIs, thresholds)
        cpm_strict = CPM(TPRs, FPIs)
        rcpm_strict = RCPM(TPRs, FPIs)


        if fauc_strict > best_value:
            best_value = fauc_strict
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    #print("epoch:{}/{}  train_loss:{:.4f}  val_loss:{:.4f}  val_mDice:{:.4f}  val_fauc:{:.4f}  val_cpm:{:.4f}  val_rcpm:{:.4f}  val_faucStrict:{:.4f}  val_cpmStrict:{:.4f}  val_rcpmStrict:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, mdice, fauc, cpm, rcpm, fauc_strict, cpm_strict, rcpm_strict))
    print("epoch:{}/{}  train_loss:{:.4f}   val_fauc:{:.4f}  val_cpm:{:.4f}  val_rcpm:{:.4f}  val_faucStrict:{:.4f}  val_cpmStrict:{:.4f}  val_rcpmStrict:{:.4f}".format(epoch + 1, num_epoch, train_loss, fauc, cpm, rcpm, fauc_strict, cpm_strict, rcpm_strict))

print("best_faucStrict:{:.4f}   (epoch:{})".format(best_value, best_epoch + 1))

#save the model since we might use it later
torch.save(best_model, saveModelPath) #best_model