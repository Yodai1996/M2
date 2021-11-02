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

from utils import train, valid, preprocess_df, collate_fn, visualize, MyDataset, mIoU, mAP, returnFnIndices

args = sys.argv
# trainPath, validPath, trainBbox, validBbox, modelName = args[1], args[2], args[3], args[4], args[5],
# num_epoch = int(args[6])
# batch_size = int(args[7])
# numSamples = int(args[8])
# pretrained = args[9]

trainPath, validPath, testPath, trainBbox, validBbox, testBbox, modelName = args[1], args[2], args[3], args[4], args[5], args[6], args[7]
num_epoch = int(args[8])
batch_size = int(args[9])
numSamples = int(args[10])
pretrained = args[11]
ver = int(args[12])
trainDir = "/lustre/gk36/k77012/M2/data/{}/".format(trainPath)
validDir = "/lustre/gk36/k77012/M2/data/{}/".format(validPath)
testDir = "/lustre/gk36/k77012/M2/data/{}/".format(testPath)
saveDir = "/lustre/gk36/k77012/M2/result/{}_{}_{}_batch{}_epoch{}_{}_{}/".format(trainPath, validPath, modelName, batch_size, num_epoch, pretrained, ver) #saveDir = "/lustre/gk36/k77012/M2/result/{}_{}_{}_batch{}_epoch{}/".format(trainPath, validPath, modelName, batch_size, num_epoch)
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

num_classes = 2 #(len(classes)) + 1
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

# training
optimizer = optim.Adam(model.parameters(), lr=lr)

best_miou, best_map = 0, 0
best_epoch = 0
best_miou_model = None

for epoch in range(num_epoch):

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

        if miou > best_miou:
            best_miou = miou
            best_miou_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        best_map = max(best_map, map)

    print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mAP:{:.4f}   test_loss:{:.4f}  test_mIoU:{:.4f}  test_mAP:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, map, test_loss, testmiou, testmap))
    #print("epoch:{}/{}  train_loss:{:.4f}  valid_loss:{:.4f}  valid_mIoU:{:.4f}  valid_mAP:{:.4f}".format(epoch + 1, num_epoch, train_loss, valid_loss, miou, map))

print("best_mIoU:{:.3f} (epoch:{}),  best_mAP:{:.3f}".format(best_miou, best_epoch + 1, best_map))

#save the model since we might use it later
PATH = "/lustre/gk36/k77012/M2/model/model_FN_search_{}".format(ver)
torch.save(best_miou_model, PATH) #best_miou_model
model.load_state_dict(torch.load(PATH)) #visualizationのときにもこのbest modelを用いることにする。

#modify and redefine again to use in visualization
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,  collate_fn=collate_fn)


#search FN indices and save
with torch.no_grad():
    FN_indices = returnFnIndices(validloader, model, numDisplay, thres=0)

with open('/lustre/gk36/k77012/M2/FN_indices_among1176_{}.csv'.format(ver), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(FN_indices)
f.close()


# visualization of training, validation and testing
visualize(model, trainloader, df, numSamples, saveDir + "train/", numDisplay)

#FN_indicesだけでよいため、無駄が発生するが、一応全部保存しておく。
numSamples = 1176
visualize(model, validloader, df_valid, numSamples, saveDir + "valid/", numDisplay)
visualize(model, validloader, df_valid, numSamples, saveDir + "original/", numDisplay=0) #gt_bboxだけにする。

