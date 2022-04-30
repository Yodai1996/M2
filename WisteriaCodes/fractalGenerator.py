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
    # decide the position where the pseudo pneumonia lesion is put on, avoiding too white positions
    size = img.size[0]  # 1024

    cnt = 0
    while (1):
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
                    s += img.getpixel((x, y))

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

    #make a white circle on the canvas
    mask_im = Image.new("L", (canvasSize, canvasSize), 0)
    draw = ImageDraw.Draw(mask_im)
    draw.ellipse((center - r, center - r, center + r, center + r), fill=255)

    #make the circle blured and scaled
    mask = np.array(mask_im)
    scaledMask = mask * scale
    scaledMask = scalingAndBluring(scaledMask, r + 1, center, smoothArea)  #we use r+1 instead of r to delete noise around the edges

    #apply an affine transformation randomly
    roi_points = [(0, 0), (canvasSize, 0), (canvasSize, canvasSize), (0, canvasSize)]
    x1, y1, x2, y2 = [int(2 * r * random.random() + center - r) for _ in range(4)] #points are chosen randomly inside the circle
    from_points = [(x1, y1)]
    to_points = [(x2, y2)]
    from_points = np.concatenate((roi_points, from_points))
    to_points = np.concatenate((roi_points, to_points))
    affin = skimage.transform.PiecewiseAffineTransform()
    success = affin.estimate(to_points, from_points)
    assert(success == True)
    transformedMask = skimage.transform.warp(scaledMask, affin)  # [0,scale)
    normalizedTransformedMask = transformedMask / 256

    #### make the bbox of the transformed mask
    x = min(np.where(sum(transformedMask) >= 1)[0])
    xmax = max(np.where(sum(transformedMask) >= 1)[0])
    y = min(np.where(sum(transformedMask.T) >= 1)[0])
    ymax = max(np.where(sum(transformedMask.T) >= 1)[0])
    width = xmax - x
    height = ymax - y

    #generate fractal perlin noise
    shape = (lacunarity**(octaves - 1)) * res
    np.random.seed(index)
    noise = generate_fractal_noise_2d(shape=(shape, shape), res=(res, res), octaves=octaves, persistence=persistence, lacunarity=lacunarity)

    noise = (noise+1)/2  #normalize between 0~1

    #resize the image
    resized_noise = resize(noise, (height + 1, width + 1))
    pneumonia = resized_noise * normalizedTransformedMask[y:ymax + 1, x:xmax + 1]

    # decide the position where the pseudo pneumonia is put on, avoiding the too white position
    centX, centY = lungPosition(newImg, pneumonia, width, height, margin, too_white, iter_cnt, increment)

    #put the pseudo pneumonia on that position
    for j in range(height):
        for i in range(width):
            x, y = int(centX - width // 2 + i), int(centY - height // 2 + j)
            inPixel = newImg.getpixel((x, y))
            pneumoniaPixel = pneumonia[j][i]
            if pneumoniaPixel > 0:
                outPixel = inPixel * (1 - pneumoniaPixel) + 255 * pneumoniaPixel #apply X-ray principle
                newImg.putpixel((x, y), int(outPixel))
                segMaskImg.putpixel((x, y), 255) #make a segmentation mask

    left, top, right, bottom = centX - width / 2, centY - height / 2, centX + width / 2, centY + height / 2

    return newImg, segMaskImg, left, top, right, bottom, centX, centY, r


def make_abnormals(ll, normal_dir, abnormal_dir, segMask_dir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea):

    #How many lesions are inserted into a normal image is determined by the following probability (based on the observation of #small-lesions in 101 dataset)
    prob1 = 0.93
    prob2 = 0.04
    prob3 = 0.03

    parameterInfo = []
    parameterInfo.append(["file", "index", "lb", "ub", "res", "octaves", "persistence", "lacunarity", "scale", "smoothArea", "cent_x", "cent_y", "r"])

    bboxInfo = []
    bboxInfo.append(["file", "left1", "top1", "right1", "bottom1", "left2", "top2", "right2", "bottom2", "left3", "top3", "right3", "bottom3", "left4", "top4", "right4", "bottom4"])

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

        #make abnormals
        #make the first lesion
        newImg, segMaskImg, left1, top1, right1, bottom1, centX, centY, r = make_abnormal(newImg, segMaskImg, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
        parameterInfo.append([file, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, centX, centY, r])
        index += 1
        if prob1 < prob:
            #second
            newImg, segMaskImg, left2, top2, right2, bottom2, centX, centY, r = make_abnormal(newImg, segMaskImg, index,
                                                                                              lb, ub, res, octaves,
                                                                                              persistence, lacunarity,
                                                                                              scale, smoothArea)
            parameterInfo.append([file, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, centX, centY, r])
            index += 1
            if prob1+prob2 < prob:
                #third
                newImg, segMaskImg, left3, top3, right3, bottom3, centX, centY, r = make_abnormal(newImg, segMaskImg,
                                                                                                  index,
                                                                                                  lb, ub, res, octaves,
                                                                                                  persistence,
                                                                                                  lacunarity,
                                                                                                  scale, smoothArea)
                parameterInfo.append([file, index, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea, centX, centY, r])
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

    #deal with both indexing formats
    if len(ll) <= 2:
        ll = ll[0]
    else:
        ll = [ll[i][0] for i in range(len(ll))]


    if len(args)>9:
        #values are directly given from command line
        lb=20
        ub=75
        res=int(args[7])
        octaves=5
        persistence=float(args[8])
        lacunarity=int(args[9])
        scale=float(args[10])
        smoothArea=float(args[11])

    elif len(args)<=8:
        #values are normalized and given through a text file
        bufText=args[7]

        # read the recommended next values from Gaussian Process.
        fileHandle = open(bufText, "r")
        lineList = fileHandle.readlines()
        fileHandle.close()
        last_lines = lineList[-1]

        if last_lines[-1] == "\n":
            last_lines = last_lines[:-2]

        values = last_lines.split(",")
        values = [float(i) for i in values]

        # decode the parameters from the normalized values
        lb = 20
        ub = 75
        octaves = 5
        res = int(2 + 4 * values[0])
        persistence = 0.2 + 0.8 * values[1]  # [0.2,1]
        lacunarity = int(2 + 3 * values[2])
        scale = 0.1 + 0.9 * values[3]  # [0.1,1]
        smoothArea = 0.2 + 0.6 * values[4]  # [0.2,0.8]

    else:
        print("arguments error happening")
        exit()

    #make abnormal data
    print("begin to make abnormal images")
    parameterInfo, bboxInfo = make_abnormals(ll, normalDir, abnormalDir, segMaskDir, lb, ub, res, octaves, persistence, lacunarity, scale, smoothArea)
    print("finish making abnormal images")

    #store the information about parameters to reproduce the same images
    with open(saveParaPath, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(parameterInfo)

    #store the information about bbox to use it in model learning
    with open(saveBboxPath, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(bboxInfo)
