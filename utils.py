import os
import shutil
import csv
import time
import numpy as np
import math
import sys
import cv2
from PIL import Image,ImageFilter
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from os.path import abspath
from prettytable import PrettyTable

curr = os.path.dirname(__file__)

def build_scaled(imgFilePath, newImgPath):
    size = 128, 128
    print('Copying images to new location...')

    for root, dirs, files in os.walk(imgFilePath):
        for idx, file in enumerate(files):
            fileName = file.split('.')[0]
            sys.stdout.flush()
            if not os.path.exists(newImgPath):
                os.makedirs(newImgPath)
            shutil.copy(os.path.join(root,file), newImgPath)
            sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
            sys.stdout.flush()
    sys.stdout.write("\b\nDone\n")

    print('Scaling test images to %s... and deleting old ones' % (str(size)))
    for root, dirs, files in os.walk(newImgPath):
        for idx, file in enumerate(files):
            outfile = 'ISIC_' + file if not 'ISIC_' in file else file
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


def crop(imgFilePath, newImgPath):
    print('Cropping images...')
    for root, dirs, files in os.walk(imgFilePath):
        for idx, file in enumerate(files):

            if not os.path.exists(newImgPath):
                os.makedirs(newImgPath)
            # if idx == 0:

            new_width = 75
            new_height = 75

            im = Image.open(os.path.join(imgFilePath, file))
            width, height = im.size
            left = int(np.floor((width - new_width)/2))
            top =  int(np.floor((height - new_height)/2))
            right = int(np.floor((width + new_width)/2))
            bottom = int(np.floor((height + new_height)/2))
            
            # Cropped image of above dimension
            # (It will not change orginal image)
            im1 = im.crop((left, top, right, bottom))
            sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
            sys.stdout.flush()
            # Shows the image in image viewer
            im1.save(os.path.join(newImgPath, file))
    sys.stdout.write("\b\nDone\n")



def dbc(img,s):
    (width, height) = img.size
    # check width == height
    assert(width == height)
    pixel = img.load()
    M = width
    # grid size must be bigger than 2 and least than M/2
    G = 256
    assert(s >= 2)
    assert(s <= M//2)
    ngrid = math.ceil(M / s)
    h = G*(s / M) # box height
    nr = np.zeros((ngrid,ngrid), dtype='int32')
    
    for i in range(ngrid):
        for j in range(ngrid):
            maxg = 0
            ming = 255
            for k in range(i*s, min((i+1)*s, M)):
                for l in range(j*s, min((j+1)*s, M)):
                    if pixel[k, l] > maxg:
                        maxg = pixel[k, l]

                    if pixel[k, l] < ming:
                        ming = pixel[k, l]
                        
            nr[i,j] = math.ceil(maxg/h) - math.ceil(ming/h) + 1

    Ns = 0
    N = np.size(nr)
    for i in range(ngrid):
        for j in range(ngrid):
            Ns += nr[i, j]
    return Ns, N, nr

def rgb(imgFilePath, newImgPath):
    t = PrettyTable(['Nr. crt.', 'Image', 'Canal', 'Ns', 'procent_box (r)', 'factor_divizare (S)', 'fractal_dim', 'lacunaritate', 'nr_boxes'])
    print('Splitting into r, g, b then calculate lacunarity and fractal dimension...')
    b_path = os.path.join(newImgPath, r'b')
    g_path = os.path.join(newImgPath, r'g')
    r_path = os.path.join(newImgPath, r'r')

    if not os.path.exists(newImgPath):
        os.makedirs(newImgPath)

    if not os.path.exists(b_path):
        os.makedirs(b_path)
    if not os.path.exists(g_path):
        os.makedirs(g_path)
    if not os.path.exists(r_path):
        os.makedirs(r_path)

    for root, dirs, files in os.walk(imgFilePath):
        for idx, file in enumerate(files):

            # if idx == 0:
            img = cv2.imread(os.path.join(imgFilePath, file))
            r, g, b = cv2.split(img)

            cv2.imwrite(os.path.join(b_path, file), b)
            cv2.imwrite(os.path.join(g_path, file), g)
            cv2.imwrite(os.path.join(r_path, file), r)

            Ns, procent_box, factor_divizare, fractal_dim, lacunaritate, nr_boxes = calc_lacunarity_and_fractal_dim(os.path.join(r_path, file))
            t.add_row([idx, file, 'R', Ns, round(procent_box, 5), round(factor_divizare, 5), round(fractal_dim, 5), round(lacunaritate, 5), nr_boxes])
            Ns, procent_box, factor_divizare, fractal_dim, lacunaritate, nr_boxes = calc_lacunarity_and_fractal_dim(os.path.join(g_path, file))
            t.add_row(['', file, 'G', Ns, round(procent_box, 5), round(factor_divizare, 5), round(fractal_dim, 5), round(lacunaritate, 5), nr_boxes])
            Ns, procent_box, factor_divizare, fractal_dim, lacunaritate, nr_boxes = calc_lacunarity_and_fractal_dim(os.path.join(b_path, file))
            t.add_row(['', file, 'B', Ns, round(procent_box, 5), round(factor_divizare, 5), round(fractal_dim, 5), round(lacunaritate, 5), nr_boxes])
            t.add_row(['-', '-', '-', '-', '-', '-', '-', '-', '-'])
            sys.stdout.write("\bCurrent progress: %s %%\r" % (str(math.ceil(idx / len(files) * 100))))
            sys.stdout.flush()
    sys.stdout.write("\b\nDone\n")
    print(t) 


def calc_lacunarity_and_fractal_dim(path):
    
    image = Image.open(path) # Brodatz/D1.gif
    image = image.convert('L')
    (imM, _) = image.size

    # calculate Nr and r
    Nr = []
    r = []
    
    a = 2
    b = imM//2
    nval = 20
    lnsp = np.linspace(1,math.log(b,a),nval)
    sval  = a**lnsp
    L = {}
    for S in sval:#range(2,imM//2,(imM//2-2)//100):
        Ns, N, nr = dbc(image, int(S))
        Nr.append(Ns)
        R = S/imM
        r.append(S)
        temp = 0
        L[int(S)] = 0
        N = int(math.sqrt(N))
        for i in range(N):
            for j in range(N):
                count = 0
                for k in range(N):
                    for l in range(N):
                        if nr[k][l] == nr[i][j]:
                            count = count + 1
                prob = count / (N*N)
                L[int(S)] = L[int(S)] + (i*j)**2 * prob
                temp = temp + i * prob
        L[int(S)] = L[int(S)] / (temp**2)
        # t.add_row([img_name, Ns, R, S, L[int(S)], N*N])

    # calculate log(Nr) and log(1/r)    
    y = np.log(np.array(Nr))
    x = np.log(1/np.array(r))
    (D, b) = np.polyfit(x, y, deg=1)

    # search fit error value
    N = len(x)
    Sum = 0
    for i in range(N):
        Sum += (D*x[i] + b - y[i])**2
        
    errorfit = (1/N)*math.sqrt(Sum/(1+D**2))

    # # figure size 10x5 inches
    # plt.figure(1,figsize=(10,5)).canvas.set_window_title('Fractal Dimension Calculate')
    # plt.subplots_adjust(left=0.04,right=0.98)
    # plt.subplot(121)
    # plt.title(path)
    # plt.imshow(image)
    # plt.axis('off')


    # plt.subplot(122)  
    # plt.title('Fractal dimension = %f\n Fit Error = %f' % (D,errorfit))

    # plt.plot(x, y, 'ro',label='Calculated points')
    # plt.plot(x, D*x+b, 'k--', label='Linear fit' )
    # plt.legend(loc=4)
    # plt.xlabel('log(1/r)')
    # plt.ylabel('log(Nr)')
    # plt.show()

    return Ns, R, S, D, L[int(S)], N*N