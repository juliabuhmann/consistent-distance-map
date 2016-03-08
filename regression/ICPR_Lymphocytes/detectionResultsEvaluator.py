# Author: Florian Jug <jug@mpi-cbg.de>

import sys
import os
from time import strftime

import scipy
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from tifffile import *

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from skimage import measure


def computeDetections( imgD, threshold, lessEqual=True ):
    """Threshold given image and return the COM of each connected component.
    """
    if lessEqual:
        imgT = imgD<=threshold
    else:
        imgT = imgD>threshold
    imgT = imgT.astype(np.int8)
    labels = measure.label(imgT, background=0)

    centroids = []
    props = measure.regionprops(labels)
    for prop in props:
        centroids.append(prop.centroid)

    return centroids, imgT


# -------------------------------------------------------------------------------
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN
# -------------------------------------------------------------------------------
PLOT = False
maxMatchingDist = 6
thresholdRange = range(1,12)

pathRegressionImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_ALL'
pathGtDetectionImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth'

fnRegressionImages = [ os.path.join(pathRegressionImages, fn) for fn in os.listdir(pathRegressionImages) if fn.endswith('.tif') ]
fnGtDetectionImages = [ os.path.join(pathGtDetectionImages, fn) for fn in os.listdir(pathGtDetectionImages) if fn.endswith('.tif') ]

results = []
for threshold in thresholdRange:
    FP = 0 # false positives
    TP = 0 # true positives
    NM = 0 # not matched GT points
    for i, fnR in enumerate(fnRegressionImages[0:]):
        imgR = imread(fnR)
        detections, imgT = computeDetections( imgR, threshold )

        fnG = fnGtDetectionImages[i]
        imgG = imread(fnG)[:,:,1] # GT is brainfucked, being RGB, havind detections in green channel
        gtDetections, imgD = computeDetections( imgG, 0.5, lessEqual=False )

        FPi = 0 # false positives
        TPi = 0 # true positives
        NMi = 0 # not matched GT points
        gtMatched = np.ones(len(gtDetections))<.5 #gives all False
        for j, det in enumerate(detections):
            matchers = [] #collect potentially matching points
            minDistFound = imgG.size
            for k, gtDet in enumerate(gtDetections):
                if gtMatched[k]: continue
                X = np.array([det, gtDet])
                x = scipy.spatial.distance.pdist(X, 'euclidean')[0]
                if x<maxMatchingDist:
                    matchers.append([k, x])
                    if minDistFound > x:
                        minDistFound = x
            for k, x in matchers:
                if x == minDistFound:
                    gtMatched[k] = True
                    TPi += 1
            if len(matchers)==0:
                FPi += 1
        NMi = np.sum(gtMatched==False) #count unmatched GT detections

        # print TPi, FPi, NMi, ' -- ',
        # print len(detections), len(gtDetections)

        TP += TPi
        FP += FPi
        NM += NMi

        if PLOT:
            pl.imshow(imgT, interpolation='nearest')
            pl.show()

    if TP+FP > 0:
        precision = float(TP)/(TP+FP)
    else:
        precision = 0.0
    if TP+NM > 0:
        recall = float(TP)/(TP+NM)
    else:
        recall = 1.0
    print 'Threshold: ', threshold, ' (Pr,Re) = ', precision, recall
    results.append([threshold, precision, recall])

results = np.array(results)

pd = results[:,1:]
pl.plot(pd[:,1],pd[:,0], 'x-', color='black', label='OUR')
# plot icpr old state of the art
pl.plot(0.901,0.868, '*', color='red', label='icpr best')
# plot MICCAI15 data (approx)
miccai15 = np.array(
           [ [.70, .98],
             [.75, .97],
             [.80, .97],
             [.85, .96],
             [.90, .92],
             [.925,.90],
             [.95, .80] ] )
pl.plot(miccai15[:,0], miccai15[:,1], 'o-', color='blue', label='MICCAI15')

pl.plot([0,1],[0,1], '--', color='gray') #, label='y=x')

pl.legend(bbox_to_anchor=(0.05, 0.25), loc=2, borderaxespad=0.)
pl.xlabel('recall')
pl.ylabel('precision')
pl.show()
