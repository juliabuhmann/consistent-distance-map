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

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))+'/../../../'
print ROOT_PATH
sys.path.append(os.path.join(ROOT_PATH,'src','python'))
import surface_reconstruction as sr
reload(sr)

fnResult1ImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Smoothed/seg_boundaries_t0'
fnResult2ImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Predictions/seg_boundaries_t0'
fnGtImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/GroundTruthCenterlines/t0'

for i in range(60,71):
    imgResult1 = imread(fnResult1ImagePrefix+str(i)+'.tif')
    imgResult2 = imread(fnResult2ImagePrefix+str(i-60)+'.tif')
    imgGt = imread(fnGtImagePrefix+str(i)+'.tif')
    score1 = sr.score( imgResult1, imgGt, score='CC_VI')
    score2 = sr.score( imgGt, imgGt, score='CC_VI')
    print score1, ' vs ', score2

