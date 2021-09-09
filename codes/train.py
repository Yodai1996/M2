import sys
import argparse
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageDraw, ImageFilter

import torch
from torch import nn, optim, distributed as dist
import torch.nn.functional as F

from torchvision import models, transforms, datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--tp")
parser.add_argument("--vp")
parser.add_argument("--tb")
parser.add_argument("--vb")
parser.add_argument("--model")
parser.add_argument("--epoch", type=int)
parser.add_argument("--bsz", type=int)
parser.add_argument("--ns", type=int)
parser.add_argument("--pret")

args = parser.parse_args()
local_rank = args.local_rank  #used at DDP
trainPath = args.tp
validPath = args.vp
trainBbox = args.tb
validBbox = args.vb
modelName = args.model
num_epoch = args.epoch
batch_size = args.bsz
numSamples = args.ns
pretrained = args.pret

#used at DDP
torch.cuda.set_device(local_rank)  # before your code runs, set your device to local rank
dist.init_process_group(backend='nccl', init_method='env://') # distributed environment
rank = dist.get_rank() #使うかはわからんが一応取得しておこう。
world_size = dist.get_world_size()

trainDir = "/lustre/gk36/k77012/M2/data/{}/".format(trainPath)
validDir = "/lustre/gk36/k77012/M2/data/{}/".format(validPath)
#testDir = "/lustre/gk36/k77012/M2/data/{}/".format(testPath)
saveDir = "/lustre/gk36/k77012/M2/result/{}_{}_{}_batch{}_epoch{}_{}/".format(trainPath, validPath, modelName, batch_size, num_epoch, pretrained) #saveDir = "/lustre/gk36/k77012/M2/result/{}_{}_{}_batch{}_epoch{}/".format(trainPath, validPath, modelName, batch_size, num_epoch)
df = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(trainBbox))
df_valid = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(validBbox))
#df_test = pd.read_csv("/lustre/gk36/k77012/M2/{}".format(testBbox))

# hypara
originalSize = 1024
size = 300
lr = 0.0002
numDisplay = 2 #the number of predicted bboxes to display, also used when calculating mIoU.
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = preprocess_df(df, originalSize, size, trainDir)
df_valid = preprocess_df(df_valid, originalSize, size, validDir)
#df_test = preprocess_df(df_test, originalSize, size, testDir)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()  # これを挟むと自動で[0,1)に正規化される
])
trainset = MyDataset(df, transform=transform)
validset = MyDataset(df_valid, transform=transform)
#testset = MyDataset(df_test, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, sampler=DistributedSampler(trainset), num_workers=4 * torch.cuda.device_count(), pin_memory=True, collate_fn=collate_fn) #DistributedSamplerを使うとデフォでshuffle=True
#validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn) #torch.no_gradなので分散させるまでもない

#本当はshuffle=Falseでやりたい
validloader = DataLoader(validset, batch_size=batch_size, sampler=DistributedSampler(validset, shuffle=False), num_workers=4 * torch.cuda.device_count(), pin_memory=True, collate_fn=collate_fn)
#testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)

num_classes = 2 #(len(classes)) + 1
if modelName=="SSD":
    model = models.detection.ssd300_vgg16(pretrained=False, pretrained_backbone=False).to(local_rank) #.to(device)
    if pretrained=="pretrained":
        model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/ssd300_vgg16_coco-b556d3b4.pth"))
    model2 = models.detection.ssd300_vgg16(num_classes = num_classes, pretrained=False, pretrained_backbone=False)
    model.head.classification_head = model2.head.classification_head.to(local_rank) #.to(device)  #modify the head of the model
else:  #modelName=="fasterRCNN"
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False).to(local_rank) #.to(device)
    if pretrained == "pretrained":
        model.load_state_dict(torch.load("/lustre/gk36/k77012/M2/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))  # model.load_state_dict(torch.load("/lustre/gk36/k77012/faster_RCNN.pth"))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(local_rank) #.to(device)

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

#DDP
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) #, output_device=local_rank)  #output_deviceは不要かもしれない

for epoch in range(num_epoch):

    trainloader.sampler.set_epoch(epoch)  # necessary to shuffle dataset
    train_loss = train(trainloader, model, local_rank, world_size, optimizer)

    # validation
    with torch.no_grad():
        valid_loss = valid(validloader, model, local_rank, world_size)
#        test_loss = valid(testloader, model)

        #calculate performance of mean IoU
        miou = mIoU(validloader, model, local_rank, world_size, numDisplay)
        map = mAP(validloader, model, local_rank, world_size, numDisplay)
#        testmiou = mIoU(testloader, model, numDisplay)
#        testmap = mAP(testloader, model, numDisplay)

    #print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mAP:{:.4f}   test_loss:{:.4f}  test_mIoU:{:.4f}  test_mAP:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, map, test_loss, testmiou, testmap))
    if rank==0 or rank==1:
        print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mAP:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, map))


# #modify and redefine again to use in visualization
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset), num_workers=4 * torch.cuda.device_count(), pin_memory=True, collate_fn=collate_fn)
# validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset), num_workers=4 * torch.cuda.device_count(), pin_memory=True, collate_fn=collate_fn)
# #testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
#
# # visualization of training, validation and testing
# visualize(model, local_rank, trainloader, df, numSamples, saveDir + "train/", numDisplay)
# visualize(model, local_rank, validloader, df_valid, numSamples, saveDir + "valid/", numDisplay)
# #visualize(model, local_rank, testloader, df_test, numSamples, saveDir + "test/", numDisplay)
