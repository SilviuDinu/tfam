import os
from utils import build_scaled, crop, rgb

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
