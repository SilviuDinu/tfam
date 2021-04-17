import os
import shutil
import csv
import time
import math
import sys
from PIL import Image
from os.path import abspath

size = 128, 128

curr = os.path.dirname(__file__)

trainCSVPath = os.path.join(curr, '../../data/csv-data/train.csv')
testCSVPath = os.path.join(curr, '../../data/csv-data/test.csv')

trainImagesOutput = os.path.join(curr, '../../data/image-data/train-images')
testImagesOutput = os.path.join(curr, '../../data/image-data/test-images')

images_path = os.path.join(curr, '../../data/image-data/images/')

def test(csvFilePath, imgFilePath, newImgPath):
    img_list = []
    print('Searching for test images...')
    with open(csvFilePath, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for idx, rows in enumerate(csv_reader):
            if idx > 0:
                img_list.append(rows[1])
    
    print('Found %d images' % (len(img_list)))
    print('Writing test images...')

    for root, dirs, files in os.walk(imgFilePath):
        for idx, file in enumerate(files):
            fileName = file.split('.')[0]
            sys.stdout.flush()
            if fileName in img_list:
                if not os.path.exists(newImgPath):
                    os.makedirs(newImgPath)
                shutil.copy(os.path.join(root,file), newImgPath)
                sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
                sys.stdout.flush()
    sys.stdout.write("\b\nDone\n")

    print('Scaling test images to %s...' % (str(size)))
    for root, dirs, files in os.walk(newImgPath):
        for idx, file in enumerate(files):
            outfile = file.split('.')[0].split('_')[1] + ".jpg"
            try:
                im = Image.open(os.path.join(newImgPath, file))
                im = im.resize(size, Image.ANTIALIAS)
                sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
                sys.stdout.flush()
                im.save(os.path.join(newImgPath, outfile), "JPEG" ,quality=100)
                os.remove(os.path.join(newImgPath, file))
            except IOError:
                print("cannot create thumbnail for '%s'" % file)
    sys.stdout.write("\b\nDone\n")



def train(csvFilePath, imgFilePath, newImgPath):
    img_list = []
    print('Searching for train images...')
    with open(csvFilePath, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for idx, rows in enumerate(csv_reader):
            if idx > 0:
                img_list.append(rows[1])

    
    print('Found %d images' % (len(img_list)))
    print('Writing train images...')
    
    for root, dirs, files in os.walk(imgFilePath):
        for idx, file in enumerate(files):
            fileName = file.split('.')[0]
            sys.stdout.flush()
            if fileName in img_list:
                if not os.path.exists(newImgPath):
                    os.makedirs(newImgPath)
                shutil.copy(os.path.join(root,file), newImgPath)
                sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
                sys.stdout.flush()
    sys.stdout.write("\b\nDone\n")

    print('Scaling train images to %s...' % (str(size)))
    for root, dirs, files in os.walk(newImgPath):
        for idx, file in enumerate(files):
            outfile = file.split('.')[0].split('_')[1] + ".jpg"
            try:
                im = Image.open(os.path.join(newImgPath, file))
                im = im.resize(size, Image.ANTIALIAS)
                sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
                sys.stdout.flush()
                im.save(os.path.join(newImgPath, outfile), "JPEG" ,quality=100)
                os.remove(os.path.join(newImgPath, file))
            except IOError:
                print("cannot create thumbnail for '%s'" % file)
    sys.stdout.write("\b\nDone\n")


train(trainCSVPath, images_path, trainImagesOutput)
test(testCSVPath, images_path, testImagesOutput)
