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


PLOT = False


fnResult1ImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Smoothed/seg_boundaries_t0'
fnResult2ImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Predictions/seg_boundaries_t0'
fnGtImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/GroundTruthCenterlines/t0'

for i in range(60,71):
    tmp = imread(fnResult1ImagePrefix+str(i)+'.tif')
    imgResult1 = np.zeros_like(tmp)
    imgResult1[tmp==0] = 1

    tmp = imread(fnResult2ImagePrefix+str(i-60)+'.tif')
    imgResult2 = np.zeros_like(tmp)
    imgResult2[tmp==0] = 1
    
    tmp = imread(fnGtImagePrefix+str(i)+'.tif')
    imgGt = np.zeros_like(tmp)
    imgGt[tmp==0] = 1
    
    if PLOT:
        pl.subplot(2,3,1)
        pl.imshow(imgResult1, interpolation='nearest')
        pl.subplot(2,3,2)
        pl.imshow(imgResult2, interpolation='nearest')
        pl.subplot(2,3,3)
        pl.imshow(imgGt, interpolation='nearest')
        pl.subplot(2,3,4)
        pl.imshow(imgResult1+imgResult2-2*imgGt, interpolation='nearest')
        pl.subplot(2,3,5)
        pl.imshow(imgResult1-imgGt, interpolation='nearest')
        pl.subplot(2,3,6)
        pl.imshow(imgResult2-imgGt, interpolation='nearest')
        pl.show()

    score1_L1 = sr.score( imgResult1, imgGt, score='L1')
    score2_L1 = sr.score( imgResult2, imgGt, score='L1')
    score1_VI = sr.score( imgResult1, imgGt, score='CC_VI')
    score2_VI = sr.score( imgResult2, imgGt, score='CC_VI')
    print score1_L1, score1_VI, '  vs  ', score2_L1, score2_VI

