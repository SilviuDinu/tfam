import os
import shutil
import csv
import time
import math
import sys
from PIL import Image
import cv2
from os.path import abspath
from utils import build_scaled, crop, rgb, dbc

curr = os.path.dirname(__file__)


images_in = os.path.join(curr, r'data/image-data/imagini')
images_out = os.path.join(curr, r'data/image-data/images/resized')

build_scaled(images_in, images_out)

# ---------------------------------------------------------------------

images_in = images_out
images_out = os.path.join(curr, r'data/image-data/images/cropped')

crop(images_in, images_out)

# ---------------------------------------------------------------------

images_in = images_out
images_out = os.path.join(curr, r'data/image-data/images/rgb/')

rgb(images_in, images_out)
