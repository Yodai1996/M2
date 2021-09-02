import sys
import math
import copy

import numpy as np
import csv
from PIL import Image, ImageDraw, ImageFilter
import scipy.sparse as sp
import pandas as pd
from numpy.random import *

import matplotlib.pyplot as plt
import skimage.transform
from skimage.transform import resize
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)

def scalingAndBluring(scaledMask, r, center, smoothArea):
    #scale and blur the white mask
    for i in range(-1 * r-2, r+2):
        for j in range(-1 * r-2, r+2):
            dist = np.sqrt(i**2 + j**2)
            if dist <= r:
                length = 1 - dist/r
                if length <= smoothArea:
                    alpha = length / smoothArea
                    scaledMask[center+i][center+j] *= alpha
    return scaledMask

def lungPosition(img, pneumonia, width, height, margin, too_white, iter_cnt, increment):
    # decide the position where the pseudo pneumonia is put on, avoiding the too white position
    size = img.size[0]  # 1024

    cnt = 0
    while (1):
        center = rand(2) * (size - margin * 2) + margin
        centX = int(center[0])
        centY = int(center[1])

        #calculate strictly, not approximately
        s = 0
        num = 0

        for j in range(height):
            for i in range(width):
                if pneumonia[j][i]>0:
                    num += 1
                    x, y = int(centX - width // 2 + i), int(centY - height // 2 + j)
                    s += img.getpixel((x, y)) #imgPixels[y][x] でもよい

        if not s > too_white * num:
            break
        else:
            cnt += 1
            if cnt > iter_cnt:
                cnt = 0
                too_white += increment

    return centX, centY


def make_abnormal(ll, normal_dir, abnormal_dir, segMask_dir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, margin=240, too_white=90, iter_cnt=20, increment=1, canvasSize = 450):
    center = canvasSize // 2  # the center point is (canvasSize//2, canvasSize//2)

    parameterInfo = []
    parameterInfo.append(["file", "index", "lb", "ub", "res", "octaves", "persistence", "lacunarity", "scale", "smoothArea", "cent_x", "cent_y", "r"])

    bboxInfo = []
    bboxInfo.append(["file", "left", "top", "right", "bottom"]) #x,y,xmax,ymax

    for index, file in enumerate(ll):
        #we use the index for the seed of noise

        r = (ub - lb) * rand() + lb
        r = int(r)

        #### make the white circle on the canvas
        mask_im = Image.new("L", (canvasSize, canvasSize), 0)
        draw = ImageDraw.Draw(mask_im)
        draw.ellipse((center - r, center - r, center + r, center + r), fill=255)

        #### make the circle blured and scaled
        mask = np.array(mask_im)
        scaledMask = mask * scale
        scaledMask = scalingAndBluring(scaledMask, r + 1, center, smoothArea)  #to delete noise we use r+1 instead of r
        pilScaledMask = Image.fromarray(scaledMask) # translate np.araray into PIL object
        pilScaledMask = pilScaledMask.convert("L")

        #### apply affine transformation randomly
        roi_points = [(0, 0), (canvasSize, 0), (canvasSize, canvasSize), (0, canvasSize)]
        x1, y1, x2, y2 = (2 * r * rand(4) + center - r) #円のbbox内からランダムに選ばれるというモデルでまず試してみる。
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        from_points = [(x1, y1)]
        to_points = [(x2, y2)]
        from_points = np.concatenate((roi_points, from_points))
        to_points = np.concatenate((roi_points, to_points))
        affin = skimage.transform.PiecewiseAffineTransform()
        success = affin.estimate(to_points, from_points)
        assert(success == True)
        # normalizedTransformedMask = skimage.transform.warp(pilScaledMask, affin)  # [0,scale)
        # transformedMask = np.array(normalizedTransformedMask * 255., dtype=np.uint8)  # [0, 255*scale]
        transformedMask = skimage.transform.warp(scaledMask, affin)  # [0,scale)
        normalizedTransformedMask = transformedMask / 256

        #### make the bbox of the transformed mask
        x = min(np.where(sum(transformedMask) >= 1)[0])
        xmax = max(np.where(sum(transformedMask) >= 1)[0])
        y = min(np.where(sum(transformedMask.T) >= 1)[0])
        ymax = max(np.where(sum(transformedMask.T) >= 1)[0])
        width = xmax - x
        height = ymax - y

        #### generate fractal perlin noise
        shape = (lacunarity**(octaves - 1)) * res
        np.random.seed(index)
        noise = generate_fractal_noise_2d(shape=(shape, shape), res=(res, res), octaves=octaves, persistence=persistence, lacunarity=lacunarity)

        #normalizing and scaling
        noise = (noise+1)/2  #0~1に正規化
        noise *= scale

        #resize the image
        resized_noise = resize(noise, (height + 1, width + 1))
        pneumonia = resized_noise * normalizedTransformedMask[y:ymax + 1, x:xmax + 1]

        #open the normal image
        img = Image.open(normal_dir + "/" + file)
        newImg = img.copy() #copy
        assert(img.size[0]==img.size[1])
        size = img.size[0]
        segMaskImg = Image.new("L", (img.size[0], img.size[1]), 0)

        # decide the position where the pseudo pneumonia is put on, avoiding the too white position
        centX, centY = lungPosition(img, pneumonia, width, height, margin, too_white, iter_cnt, increment)

        #put the pseudo pneumonia on that position
        for j in range(height):
            for i in range(width):
                x, y = int(centX - width // 2 + i), int(centY - height // 2 + j)
                inPixel = img.getpixel((x, y)) #imgPixels[y][x]でもよい
                pneumoniaPixel = pneumonia[j][i]
                if pneumoniaPixel > 0:
                    outPixel = inPixel * (1 - pneumoniaPixel) + 255 * pneumoniaPixel #X線レントゲン原理
                    newImg.putpixel((x, y), int(outPixel))
                    segMaskImg.putpixel((x, y), 255) #make segmentation mask

        newImg.save(abnormal_dir + "/" + file)
        segMaskImg.save(segMask_dir + "/" + file)

        #index means the seed for the noise
        parameterInfo.append([file, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, centX, centY, r]) #ほかのパラメタも一応入れとく。
        bboxInfo.append([file, centX - width // 2, centY - height // 2, centX + width // 2, centY + height // 2])

    return parameterInfo, bboxInfo



if __name__ == '__main__':
    args = sys.argv
    train_dir = "/lustre/gk36/k77012/M2/data/train_" + args[1]
    normal_dir = train_dir + "/0_normal1000"
    abnormal_dir = train_dir + "/1_abnormal1000_" + args[1]
    segMask_dir = "/lustre/gk36/k77012/M2/data/Mask/mask_" + args[1]

    # file name index
    with open('/lustre/gk36/k77012/M2/normalIdList1000.csv', 'r') as f:
        read = csv.reader(f)
        ll = list(read)[0]
    f.close()

    lb=int(args[2])
    ub=int(args[3])
    res=int(args[4])
    octaves=int(args[5])
    persistence=float(args[6])
    lacunarity=int(args[7])
    scale=float(args[8])
    smoothArea=float(args[9])

    # make abnormal data
    print("begin to make abnormal images")
    parameterInfo, bboxInfo = make_abnormal(ll, normal_dir, abnormal_dir, segMask_dir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
    print("finish making abnormal images")

    #store the information about parameters to reproduce the same images
    with open('/lustre/gk36/k77012/M2/simDataInfo/paraInfo/parameterInfo_{}.csv'.format(args[1]), 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(parameterInfo) #instead of writerow

    #store the information about bbox to use it in model learning
    with open('/lustre/gk36/k77012/M2/simDataInfo/bboxInfo/bboxInfo_{}.csv'.format(args[1]), 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(bboxInfo) #instead of writerow
