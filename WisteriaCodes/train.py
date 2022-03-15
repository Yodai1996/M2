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

trainPath, validPath, testPath, trainBbox, validBboxName, testBboxName, modelPath, modelName = args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]
num_epoch, batch_size, numSamples = int(args[9]), int(args[10]), int(args[11])
pretrained, saveDir, saveFROCPath, version, optimizerName = args[12], args[13], args[14], args[15], args[16]

validBbox, testBbox = validBboxName + ".csv", testBboxName + ".csv"  #valid と testだけBboxNameで引数渡した.

trainDir = "/work/gk36/k77012/M2/data/{}/".format(trainPath)
validDir = "/work/gk36/k77012/M2/data/{}/".format(validPath)
testDir = "/work/gk36/k77012/M2/data/{}/".format(testPath)
df = pd.read_csv("/work/gk36/k77012/M2/{}".format(trainBbox))
df_valid = pd.read_csv("/work/gk36/k77012/M2/{}".format(validBbox))
df_test = pd.read_csv("/work/gk36/k77012/M2/{}".format(testBbox))

# hypara
originalSize = 1024
size = 300
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_df(df, originalSize, size, trainDir)
df_valid = preprocess_df(df_valid, originalSize, size, validDir)
df_test = preprocess_df(df_test, originalSize, size, testDir)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])

inDim = 3 #3
trainset = MyDataset(df, transform=transform, inDim=inDim)
validset = MyDataset(df_valid, transform=transform, inDim=inDim)
testset = MyDataset(df_test, transform=transform, inDim=inDim)
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

    if pretrained=="pretrained" or pretrained=="ImageNet": #same meaning
        pass
    elif pretrained=="BigBbox":
        loadModelPath="/work/gk36/k77012/M2/model/pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120" #とりあえず、VSGD, noise=0.01を使用すれことにする。
        model.load_state_dict(torch.load(loadModelPath))
    else:
        loadModelPath = f"/work/gk36/k77012/M2/model/pretrain/{pretrained}"  # とりあえず、VSGD, noise=0.01を使用すれことにする。
        model.load_state_dict(torch.load(loadModelPath))

    # if pretrained=="BigBbox":
    #     loadModelPath=modelPath+"pretrain/model_nonSmall_bboxInfo_655_nonSmall_bboxInfo_164_withNormal_VSGD_0.01_120" #とりあえず、VSGD, noise=0.01を使用すれことにする。
    #     model.load_state_dict(torch.load(loadModelPath))

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

for epoch in range(num_epoch):

    train_loss = train(trainloader, model, optimizer)

    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]

    #see the performance on the training dataset
    TPRs, FPIs, thresholds = FROC(trainloader, model, thresholds=thresholds) #, ignore_big_bbox=Trueは合ってもなくても同じ。そもそも大bboxはないので。
    fauc_train = FAUC(TPRs, FPIs)
    rcpm_train = RCPM(TPRs, FPIs)

    # validation
    with torch.no_grad():

        #不要だがまあ一応。
        TPRs, FPIs, thresholds = FROC(validloader, model, thresholds=thresholds)
        fauc = FAUC(TPRs, FPIs)
        rcpm = RCPM(TPRs, FPIs)

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

    print("epoch:{}/{}  tr_loss:{:.4f}   tr_fauc:{:.4f}   tr_rcpm:{:.4f}    val_fauc:{:.4f}   val_rcpm:{:.4f}   val_fauc_IB:{:.4f}  val_rcpm_IB:{:.4f}".format(epoch + 1, num_epoch, train_loss, fauc_train, rcpm_train, fauc, rcpm, fauc_IB, rcpm_IB)) #strict is deleted

print("best_fauc_IgnoreBigBbox:{:.4f}   (epoch:{})".format(best_value, best_epoch + 1))
print("test_fauc_IgnoreBigBbox:{:.4f}, at the epoch.".format(test_performance))

#save the model since we might use it later
PATH = modelPath + f"model_version{version}_{trainPath}_{validBboxName}_{pretrained}_{optimizerName}_epoch{num_epoch}"
torch.save(best_model, PATH) #best_model
model.load_state_dict(torch.load(PATH)) #visualizationのときにもこのbest modelを用いることにする。


#visualize the fROC in which bigboxes are ignored
#validation
TPRs, FPIs, thresholds = FROC(validloader, model, thresholds=thresholds, ignore_big_bbox=True)
fauc = FAUC(TPRs, FPIs)
rcpm = RCPM(TPRs, FPIs)
print("FROC_valid_ignoreBigBbox   val_fauc:{:.4f}  val_rcpm:{:.4f}".format(fauc, rcpm))
plotFROC(TPRs, FPIs, saveFROCPath + f"FROC_valid_ignoreBigBbox_{trainPath}_{validBboxName}_{testBboxName}_{pretrained}_{optimizerName}_{num_epoch}_version{version}.png")

#test
TPRs, FPIs, thresholds = FROC(testloader, model, thresholds=thresholds, ignore_big_bbox=True)
fauc = FAUC(TPRs, FPIs)
rcpm = RCPM(TPRs, FPIs)
print("FROC_test_ignoreBigBbox    val_fauc:{:.4f}   val_rcpm:{:.4f}".format(fauc, rcpm))
plotFROC(TPRs, FPIs, saveFROCPath + f"FROC_test_ignoreBigBbox_{trainPath}_{validBboxName}_{testBboxName}_{pretrained}_{optimizerName}_{num_epoch}_version{version}.png")


#modify and redefine again to use in visualization
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

# visualization of training, validation and testing
visualize(model, trainloader, df, numSamples, saveDir + "train/", thres=0.3)
visualize(model, validloader, df_valid, numSamples, saveDir + "valid/", thres=0.3)
visualize(model, testloader, df_test, numSamples, saveDir + "test/", thres=0.3)