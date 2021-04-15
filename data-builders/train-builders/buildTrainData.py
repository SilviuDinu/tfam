import os
import shutil
import csv
import torch
import torchvision
from torchvision import transforms
import time
import math
from PIL import Image
import sys
import numpy as np
from enum import Enum
from os.path import abspath


curr = os.path.dirname(__file__)

trainCSVPath = os.path.join(curr, '../../data/csv-data/train.csv')

trainImagesPath = os.path.join(curr, '../../data/image-data/train-images')
trainImagesOutput = os.path.join(curr, '../../data/image-data/scaled-train-images')


size = 128, 128


def build_train(csvFilePath, imgFilePath, newImgPath):
    raw = []
    processed = []

    with open(csvFilePath, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for idx, rows in enumerate(csv_reader):
            if idx > 0:
                name = rows[1].split('_')[1]
                status = get_status(rows[3])
                raw.append([int(name), status])

            
    for i, couple in enumerate(raw):
        for root, dirs, files in os.walk(imgFilePath):
            for idx, file in enumerate(files):
                fileName = int(file.split('.')[0])
                if fileName == couple[0]:
                    # if not os.path.exists(newImgPath):
                    #     os.makedirs(newImgPath)
                    # outfile = os.path.splitext(file)[0] + ".jpg"
                    # try:
                    #     im = Image.open(os.path.join(imgFilePath, file))
                    #     im.thumbnail(size, Image.ANTIALIAS)
                    #     sys.stdout.write("\b[Item %d of %d: %s %%]\r" % (i, len(raw), str(math.ceil(idx / len(files) * 100))))
                    #     sys.stdout.flush()
                    #     im.save(os.path.join(newImgPath, outfile), "JPEG" ,quality=100)
                    # except IOError:
                    #     print("cannot create thumbnail for '%s'" % file)
                    # print(Image.open(os.path.join(newImgPath, file)).convert('RGB'))


def get_status(status):
    if status == 'benign':
        return 0
    elif status == 'malignant':
        return 1

build_train(trainCSVPath, trainImagesPath, trainImagesOutput)