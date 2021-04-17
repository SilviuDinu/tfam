import os
import shutil
import csv
import numpy as np
import time
import math
import sys
from PIL import Image
from matplotlib.pyplot import *
import cv2
from os.path import abspath

curr = os.path.dirname(__file__)

default_path = '/Users/silviu/Desktop/projects/cercetare_sem2/data/image-data/test-images'

input_path = input("Path to images that need to be cropped: ")
output_path = os.path.join(curr, '../../data/image-data/cropped-images/')

def crop(imgFilePath, newImgPath):

    print('Searching for images to crop in given path...')

    for root, dirs, files in os.walk(imgFilePath):
        print('Found %d images' % (len(files)))
        print('Cropping images...')
        for idx, file in enumerate(files):
            fileName = file.split('.')[0]
            if not os.path.exists(newImgPath):
                os.makedirs(newImgPath)
           
             # load image
            img = cv2.imread(os.path.join(imgFilePath, file)) 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

            # threshold to get just the signature
            retval, thresh_gray = cv2.threshold(gray, thresh=20, maxval=255, type=cv2.THRESH_BINARY)

            # find where the signature is and make a cropped region
            points = np.argwhere(thresh_gray==0) # find where the black pixels are
            points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
            x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
            x, y, w, h = x-100, y-100, w+100, h+100 # make the box a little bigger
            crop = gray[y:y+h, x:x+w] # create a cropped region of the gray image

            # get the thresholded crop
            retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
            new_img =  gray = cv2.cvtColor(thresh_crop, cv2.COLOR_GRAY2RGB) # convert to grayscale


            cv2.imwrite(os.path.join(newImgPath, file), new_img)

            sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
            sys.stdout.flush()

    sys.stdout.write("\bImages saved to %s\n" % (os.path.join(curr,newImgPath)))

   


crop(input_path, output_path)