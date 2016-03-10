# -*- coding: utf-8 -*-
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
from skimage import filters
from skimage import feature
from skimage import morphology


def computeDetections( imgD, threshold, lessEqual=True ):
    """Thresholds given image and return the COM of each connected component.
    """
    if lessEqual:
        imgT = imgD<=threshold
        # imgT = filters.threshold_adaptive(-imgD, 2*threshold, 'mean')
    else:
        imgT = imgD>threshold
    imgT = imgT.astype(np.int8)
    labels = measure.label(imgT, background=0)

    centroids = []
    props = measure.regionprops(labels)
    for prop in props:
        centroids.append(prop.centroid)

    return centroids, imgT


def computeDetectionsAtMinima( imgD, threshold, kappa=1, min_component_size=0 ):
    """Thresholds given image and returns all local minima spaced further apart than κ.
    """
    imgT = imgD*(-1)+imgD.max()
    imgT = filters.gaussian_filter(imgT/imgT.max(), sigma=1.2)
    imgT[imgD>threshold]=0

    # remove small crap if wanted
    if min_component_size>0:
        labels = measure.label(np.array(imgT>0, np.int8), background=0)
        labels = morphology.remove_small_objects(labels+1, min_size=min_component_size)
        imgT[labels==0]=0

    peaks = feature.peak_local_max( imgT, min_distance=kappa, threshold_abs=0.0005, exclude_border=False, indices=True )
    return peaks, imgT


def matchDetections( detections, gtDetections, maxMatchingDist ):
    """Matches detections in given thresholded image to GT detections
    """
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
    return TPi, FPi, NMi

def computePrecisionAndRecall( TP, FP, NM, verbose=False ):
    if TP+FP > 0:
        precision = float(TP)/(TP+FP)
    else:
        precision = 0.0
    if TP+NM > 0:
        recall = float(TP)/(TP+NM)
    else:
        recall = 1.0

    if verbose:
        print '(Pr,Re) = ', precision, recall

    return precision, recall


# -------------------------------------------------------------------------------
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN
# -------------------------------------------------------------------------------
PLOT = False
maxMatchingDist = 6
thresholdRange = range(1,12)

pathRawImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/data'
# pathRegressionImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_ALL'
# pathSmoothedImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_ALL'

# pathRegressionImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_rfR_ALL'
# pathSmoothedImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_rfR_ALL'

pathRegressionImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions_rfR_noSparse_200trees'
pathSmoothedImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/smoothed_rfR_noSparse_200trees'

pathGtDetectionImages = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth'

fnRawImages = [ os.path.join(pathRawImages, fn) for fn in os.listdir(pathRawImages) if fn.endswith('.tif') ]
fnRegressionImages = [ os.path.join(pathRegressionImages, fn) for fn in os.listdir(pathRegressionImages) if fn.endswith('.tif') ]
fnSmoothedImages = [ os.path.join(pathSmoothedImages, fn) for fn in os.listdir(pathSmoothedImages) if fn.endswith('.tif') ]
fnGtDetectionImages = [ os.path.join(pathGtDetectionImages, fn) for fn in os.listdir(pathGtDetectionImages) if fn.endswith('.tif') ]

resultsR= []
resultsS= []
for threshold in thresholdRange:
    FP_R = 0 # false positives for regression
    TP_R = 0 # true positives for regression
    NM_R = 0 # not matched GT points for regression
    FP_S = 0
    TP_S = 0 # same for smoothed
    NM_S = 0
    for i, fnR in enumerate(fnRegressionImages[0:]):
        imgR = imread(fnR)
        # detectionsR, imgRT = computeDetections( imgR, threshold )
        detectionsR, imgRT = computeDetectionsAtMinima( imgR, 5, kappa=threshold, min_component_size=10 )

        fnS = fnSmoothedImages[i]
        imgS = imread(fnS)
        # detectionsS, imgST = computeDetections( imgS, threshold )
        detectionsS, imgST = computeDetectionsAtMinima( imgS, 5, kappa=threshold, min_component_size=10 )

        fnG = fnGtDetectionImages[i]
        imgG = imread(fnG)[:,:,1] # GT is brainfucked, being RGB, havind detections in green channel
        gtDetections, imgD = computeDetections( imgG, 0.5, lessEqual=False )

        fnI = fnRawImages[i]
        imgI = imread(fnI) #[:,:,1]

        TPi, FPi, NMi = matchDetections( detectionsR, gtDetections, maxMatchingDist )
        TP_R += TPi
        FP_R += FPi
        NM_R += NMi

        TPi, FPi, NMi = matchDetections( detectionsS, gtDetections, maxMatchingDist )
        TP_S += TPi
        FP_S += FPi
        NM_S += NMi

        if False: # and PLOT or (threshold==4 and i==1):
            dR = [detectionsR[:,i] for i in range(2)]
            imgRT[dR[0],dR[1]]=-2
            dS = [detectionsS[:,i] for i in range(2)]
            imgST[dS[0],dS[1]]=-2
            pl.subplot(2,3,1)
            pl.imshow(imgR, interpolation='nearest')
            pl.subplot(2,3,2)
            pl.imshow(imgRT+5*imgD, interpolation='nearest')
            pl.subplot(2,3,3)
            pl.imshow(imgI, interpolation='nearest')

            pl.subplot(2,3,4)
            pl.imshow(imgS, interpolation='nearest')
            pl.subplot(2,3,5)
            pl.imshow(imgST+5*imgD, interpolation='nearest')
            pl.subplot(2,3,6)
            pl.imshow(imgI, interpolation='nearest')

            pl.show()

    print 'Threshold: ', threshold, ' ',
    precision, recall = computePrecisionAndRecall( TP_R, FP_R, NM_R, verbose=True )
    resultsR.append([threshold, precision, recall])
    print '               ',
    precision, recall = computePrecisionAndRecall( TP_S, FP_S, NM_S, verbose=True )
    resultsS.append([threshold, precision, recall])


# plot our results
resultsR = np.array(resultsR)
resultsS = np.array(resultsS)
pdR = resultsR[:,1:]
pl.plot(pdR[:,1],pdR[:,0], 'x-', color='gray', label='OUR_reg')
pdS = resultsS[:,1:]
pl.plot(pdS[:,1],pdS[:,0], '*-', color='black', label='OUR_regseg')

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

pl.legend(bbox_to_anchor=(0.05, 0.3), loc=2, borderaxespad=0.)
pl.xlabel('recall')
pl.ylabel('precision')
pl.show()
