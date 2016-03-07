# -*- coding: utf-8 -*-

import os
import sys
import pylab as pl
import numpy as np
import skimage as ski

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian_filter
from tifffile import *
from skimage.segmentation import find_boundaries

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))+'/../../'
print ROOT_PATH
sys.path.append(os.path.join(ROOT_PATH,'src','python'))
sys.path.append(os.path.join(ROOT_PATH,'src','python','util'))
import surface_reconstruction as sr
import dist_transform as dt
reload(sr)
reload(dt)


PLOT = False
SAVE = True


foldersInput = [
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth']
foldersOutput = [
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance']


for i,folderInput in enumerate(foldersInput):
    folderOutput = foldersOutput[i]

    filelist= [fn for fn in os.listdir(folderInput) if fn.endswith('.tif')]
    for fn in filelist:
        img = np.array(imread(os.path.join(folderInput,fn)),np.bool)
        imgDist = dt.distTransform( img[:,:,1], max_dist=15 )

        if PLOT:
            pl.imshow(imgDist, interpolation='nearest')
            pl.show()

        if SAVE:
            imsave(os.path.join(folderOutput, os.path.splitext(fn)[0]+'.tif'),imgDist.astype(np.int8),compress=0)
