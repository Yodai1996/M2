import sys
import math
import copy

import numpy as np
import csv
from PIL import Image, ImageDraw, ImageFilter
import scipy.sparse as sp
import pandas as pd
from numpy.random import *
import random

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
        # center = rand(2) * (size - margin * 2) + margin
        # centX = int(center[0])
        # centY = int(center[1])
        centX = int(random.random() * (size - margin * 2) + margin)
        centY = int(random.random() * (size - margin * 2) + margin)

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

def make_abnormal(newImg, segMaskImg, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, margin=240, too_white=90, iter_cnt=20, increment=1, canvasSize = 200):

    center = canvasSize // 2  # the center point is (canvasSize//2, canvasSize//2)

    r = (ub - lb) * random.random() + lb
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
    # x1, y1, x2, y2 = (2 * r * rand(4) + center - r) #円のbbox内からランダムに選ばれるというモデルでまず試してみる。
    # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1, y1, x2, y2 = [int(2 * r * random.random() + center - r) for _ in range(4)] #円のbbox内からランダムに選ばれるというモデルでまず試してみる。
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

    #normalizing
    noise = (noise+1)/2  #0~1に正規化

    #resize the image
    resized_noise = resize(noise, (height + 1, width + 1))
    pneumonia = resized_noise * normalizedTransformedMask[y:ymax + 1, x:xmax + 1]

    # decide the position where the pseudo pneumonia is put on, avoiding the too white position
    centX, centY = lungPosition(newImg, pneumonia, width, height, margin, too_white, iter_cnt, increment)

    #put the pseudo pneumonia on that position
    for j in range(height):
        for i in range(width):
            x, y = int(centX - width // 2 + i), int(centY - height // 2 + j)
            inPixel = newImg.getpixel((x, y)) #imgPixels[y][x]でもよい
            pneumoniaPixel = pneumonia[j][i]
            if pneumoniaPixel > 0:
                outPixel = inPixel * (1 - pneumoniaPixel) + 255 * pneumoniaPixel #X線レントゲン原理
                newImg.putpixel((x, y), int(outPixel))
                segMaskImg.putpixel((x, y), 255) #make segmentation mask

    left, top, right, bottom = centX - width / 2, centY - height / 2, centX + width / 2, centY + height / 2

    return newImg, segMaskImg, left, top, right, bottom, centX, centY, r


def make_abnormals_randomly(ll, normal_dir, abnormal_dir, segMask_dir):

    #これらはfixされているので。
    lb = 20  # int(30 + 50 * values[0])  #[30,80]
    ub = 75  # int(100 + 120 * values[1]) #[100,220]
    octaves = 5  # fixed

    #病変を何個作成するか。
    prob1 = 0.93
    prob2 = 0.04
    prob3 = 0.03

    # こっちは縦につなげるか。見やすいので。
    parameterInfo = []
    parameterInfo.append(["file", "index", "lb", "ub", "res", "octaves", "persistence", "lacunarity", "scale", "smoothArea", "cent_x", "cent_y", "r"])

    bboxInfo = []
    bboxInfo.append(["file", "left1", "top1", "right1", "bottom1", "left2", "top2", "right2", "bottom2", "left3", "top3", "right3", "bottom3", "left4", "top4", "right4", "bottom4"]) #x,y,xmax,ymax #こっちは横につなげよう。無ければNoneでよい.

    index = 0

    images, newImages, maskImages = [], [], []
    for file in ll:
        img = Image.open(normal_dir + "/" + file)
        images.append(img)
        newImg = img.copy()
        newImages.append(newImg)
        segMaskImg = Image.new("L", (img.size[0], img.size[1]), 0)
        maskImages.append(segMaskImg)

    for i, file in enumerate(ll):
        newImg, segMaskImg = newImages[i], maskImages[i]

        prob = random.random()

        left2, top2, right2, bottom2 = None, None, None, None
        left3, top3, right3, bottom3 = None, None, None, None
        left4, top4, right4, bottom4 = None, None, None, None

        res = int(2 + 4 * random.random())
        persistence = 0.2 + 0.8 * random.random()  # [0.2,1]
        lacunarity = int(2 + 3 * random.random())
        scale = 0.1 + 0.9 * random.random()  # [0.1,1]
        smoothArea = 0.2 + 0.6 * random.random()  # [0.2,0.8]

        #make abnormals
        #１個目作成
        newImg, segMaskImg, left1, top1, right1, bottom1, centX, centY, r = make_abnormal(newImg, segMaskImg, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
        parameterInfo.append([file, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, centX, centY, r])  # ほかのパラメタも一応入れとく。
        index += 1
        if prob1 < prob:
            #2個目も作成
            res = int(2 + 4 * random.random())
            persistence = 0.2 + 0.8 * random.random()  # [0.2,1]
            lacunarity = int(2 + 3 * random.random())
            scale = 0.1 + 0.9 * random.random()  # [0.1,1]
            smoothArea = 0.2 + 0.6 * random.random()  # [0.2,0.8]

            newImg, segMaskImg, left2, top2, right2, bottom2, centX, centY, r = make_abnormal(newImg, segMaskImg, index,
                                                                                              lb, ub, res, octaves,
                                                                                              persistence, lacunarity,
                                                                                              scale, smoothArea)
            parameterInfo.append([file, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, centX, centY, r])  # ほかのパラメタも一応入れとく。
            index += 1
            if prob1+prob2 < prob:
                # 3個目も作成
                res = int(2 + 4 * random.random())
                persistence = 0.2 + 0.8 * random.random()  # [0.2,1]
                lacunarity = int(2 + 3 * random.random())
                scale = 0.1 + 0.9 * random.random()  # [0.1,1]
                smoothArea = 0.2 + 0.6 * random.random()  # [0.2,0.8]

                newImg, segMaskImg, left3, top3, right3, bottom3, centX, centY, r = make_abnormal(newImg, segMaskImg,
                                                                                                  index,
                                                                                                  lb, ub, res, octaves,
                                                                                                  persistence,
                                                                                                  lacunarity,
                                                                                                  scale, smoothArea)
                parameterInfo.append([file, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, centX, centY, r])  # ほかのパラメタも一応入れとく。
                index += 1

        bboxInfo.append([file, left1, top1, right1, bottom1, left2, top2, right2, bottom2, left3, top3, right3, bottom3, left4, top4, right4, bottom4])
        newImages[i], maskImages[i] = newImg, segMaskImg

    #save
    for i, file in enumerate(ll):
        newImg, segMaskImg = newImages[i], maskImages[i]
        newImg.save(abnormal_dir + "/" + file)
        segMaskImg.save(segMask_dir + "/" + file)

    return parameterInfo, bboxInfo


if __name__ == '__main__':
    args = sys.argv
    normalIdList, normalDir, abnormalDir, segMaskDir, saveParaPath, saveBboxPath = args[1], args[2], args[3], args[4], args[5], args[6]

    # file name index
    with open(normalIdList, 'r') as f:
        read = csv.reader(f)
        ll = list(read)
    f.close()

    # 一応両方バージョン書いとく。
    if len(ll) <= 2: #１行版
        ll = ll[0] # [ll[0][i] for i in range(len(ll))]
    else: #改行版
        ll = [ll[i][0] for i in range(len(ll))]

    # make abnormal data
    print("begin to make abnormal images")
    #parameterInfo, bboxInfo = make_abnormal(ll, normal_dir, abnormal_dir, segMask_dir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
    #parameterInfo, bboxInfo = make_abnormals(ll, normalDir, abnormalDir, segMaskDir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
    parameterInfo, bboxInfo = make_abnormals_randomly(ll, normalDir, abnormalDir, segMaskDir)
    print("finish making abnormal images")

    #store the information about parameters to reproduce the same images
    with open(saveParaPath, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(parameterInfo) #instead of writerow

    #store the information about bbox to use it in model learning
    with open(saveBboxPath, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(bboxInfo) #instead of writerow
