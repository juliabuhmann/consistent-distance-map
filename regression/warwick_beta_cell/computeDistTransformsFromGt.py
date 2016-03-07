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
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/manual_annotation/m_luxian/', 
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/manual_annotation/m_sylvie/', 
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/manual_annotation/m_unionGT', 
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/manual_annotation/m_yifang']
foldersOutput = [
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/ground_truth_distmaps/m_luxian',
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/ground_truth_distmaps/m_sylvie',
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/ground_truth_distmaps/m_unionGT',
        '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/ground_truth_distmaps/m_yifang']


for i,folderInput in enumerate(foldersInput):
    folderOutput = foldersOutput[i]

    filelist= [fn for fn in os.listdir(folderInput) if fn.endswith('.png')]
    for fn in filelist:
        img = np.array(pl.mpl.image.imread(os.path.join(folderInput,fn)),np.bool)
        imgDist = dt.distTransform( img, max_dist=15 )

        if PLOT:
            pl.imshow(imgDist, interpolation='nearest')
            pl.show()

        if SAVE:
            imsave(os.path.join(folderOutput, os.path.splitext(fn)[0]+'.tif'),imgDist.astype(np.int8),compress=0)
