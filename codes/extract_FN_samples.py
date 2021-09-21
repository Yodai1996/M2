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

args = sys.argv
ver = int(args[1])

df = pd.read_csv('/lustre/gk36/k77012/M2/abnormal1176_bboxinfo_{}.csv'.format(ver))

with open('/lustre/gk36/k77012/M2/FN_indices_among1176_{}.csv'.format(ver), 'r') as f:
    read = csv.reader(f)
    indices = list(read)[0]
f.close()

FN_files = []
for ind in indices:
    ind = int(ind)
    FN_files.append(df["file"][ind])

#save data
###epoch 40 に変えよ！
resultPath="/lustre/gk36/k77012/M2/result/AbnormalDir5880_AbnormalDir5880_SSD_batch64_epoch40_pretrained_{}".format(ver)
savePath="/lustre/gk36/k77012/M2/FN_samples"
for file in FN_files:
    img = Image.open(resultPath + "/valid/" + file)
    img.save(savePath + "/Images_with_gt_and_inference/" + file)

    img = Image.open(resultPath + "/original/" + file)
    img.save(savePath + "/Images_with_gt/" + file)

    img = Image.open("/lustre/gk36/k77012/M2/data/AbnormalDir5880/" + file)
    img.save(savePath + "/Original_images/" + file)

#save the file names
with open('/lustre/gk36/k77012/M2/FN_files_among1176_{}.csv'.format(ver), 'w', newline='') as f:
    writer = csv.writer(f)
    for v in FN_files:
        writer.writerow([v])
f.close()

