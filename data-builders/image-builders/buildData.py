import os
import shutil
import csv
import time
import sys
from os.path import abspath

curr = os.path.dirname(__file__)
dirname = os.path.join(curr, )

trainCSVPath = os.path.join(curr, '../../data/csv-data/train.csv')
testCSVPath = os.path.join(curr, '../../data/csv-data/test.csv')

trainImagesOutput = os.path.join(curr, '../../data/image-data/train-images')
testImagesOutput = os.path.join(curr, '../../data/image-data/test-images')

images_path = os.path.join(curr, '../../data/image-data/images/')

def test(csvFilePath, imgFilePath, newImgPath):
    img_list = []
    print('Writing test images...')
    with open(csvFilePath, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for rows in csv_reader:
            img_list.append(rows[1])
    
    for root, dirs, files in os.walk(imgFilePath):
        for idx, file in enumerate(files):
            fileName = file.split('.')[0]
            sys.stdout.flush()
            if fileName in img_list:
                shutil.copy(os.path.join(root,file), newImgPath)
                sys.stdout.flush()
                sys.stdout.write("\bCurrent progress: %s %%\r" % (str(int((idx / len(files)) * 100))))
                sys.stdout.flush()
    sys.stdout.write("\bDone")


def train(csvFilePath, imgFilePath, newImgPath):
    img_list = []
    print('Writing train images...')
    with open(csvFilePath, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for rows in csv_reader:
            img_list.append(rows[1])
    
    for root, dirs, files in os.walk(imgFilePath):
        for idx, file in enumerate(files):
            fileName = file.split('.')[0]
            sys.stdout.flush()
            if fileName in img_list:
                shutil.copy(os.path.join(root,file), newImgPath)
                sys.stdout.flush()
                sys.stdout.write("\bCurrent progress: %s %%\r" % (str(int((idx / len(files)) * 100))))
                sys.stdout.flush()
    sys.stdout.write("\bDone")


train(trainCSVPath, images_path, trainImagesOutput)
test(testCSVPath, images_path, testImagesOutput)
