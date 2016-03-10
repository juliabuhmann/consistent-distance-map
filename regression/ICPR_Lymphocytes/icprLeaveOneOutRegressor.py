# -*- coding: utf-8 -*-
# Author: Florian Jug <jug@mpi-cbg.de>

import sys
import os
from time import strftime

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from tifffile import *

import pylab
from sklearn.externals import joblib

def train( featureStackFilenames, groundTruthFilenames, subsample_rate=1.0, addAllBelowEqual=-1, numTrees=200 ):
  assert len(featureStackFilenames)==len(groundTruthFilenames)

  for i,fnF in enumerate(featureStackFilenames):
    fnG = groundTruthFilenames[i]

    # Images have size 3 x size_X x size_Y x N_features_per_channel. Reshape to size_X x size_Y x 3 x N_features_per_channel
    imF = np.swapaxes(np.swapaxes(imread(fnF),0,2),0,1) 
    imG = imread(fnG)

    # print fnF
    # print imF.shape
    # print fnG
    # print imG.shape
    n_features = imF.shape[3]*imF.shape[2]
    n_y = imF.shape[0]
    n_x = imF.shape[1]
    n_pixels = n_y*n_x

    X_i = imF.reshape( (n_pixels, n_features) )
    y_i = imG.reshape( (n_pixels) )

    print 'Max. samples:   ', y_i.size

    # Subsample to save time and memory
    elements_to_take = np.random.rand(y_i.size)<subsample_rate
    X_i = X_i[elements_to_take]
    y_i = y_i[elements_to_take]

    # add all samples below and euqal to a certain GT value
    X_i_all = X_i[y_i<=addAllBelowEqual]
    y_i_all = y_i[y_i<=addAllBelowEqual]
    X_i=np.concatenate( (X_i,X_i_all) ,axis=0 )
    y_i=np.concatenate( (y_i,y_i_all) ,axis=0 )

    print 'Samples to add: ', y_i.size

    if 'X' in locals():
      X = np.concatenate((X,X_i))
      y = np.concatenate((y,y_i))
    else:
      X = X_i
      y = y_i

  # params = {'n_estimators': numTrees, 'max_depth': 4, 'min_samples_split': 1,
            # 'learning_rate': 0.01, 'loss': 'lad', 'verbose': 3}
  # clf = ensemble.GradientBoostingRegressor(**params)
  # clf.fit(X, y)
  # return clf, X, y

  params = {'n_estimators': numTrees, 'max_depth': None, 'min_samples_split': 2,
            'n_jobs': 8, 'verbose': 3}
  rfr = ensemble.RandomForestRegressor(**params)
  rfr.fit(X, y)
  return rfr, X, y


def regress ( regressor, featureStackFilenames ):
  predictedImages = []

  for i,fnF in enumerate(featureStackFilenames):
    print ' >> Starting to regress image ', i+1, ' of ', len(featureStackFilenames)

    imF = np.swapaxes(np.swapaxes(imread(fnF),0,2),0,1) 
    
    n_features = imF.shape[3]*imF.shape[2]
    n_y = imF.shape[0]
    n_x = imF.shape[1]
    n_pixels = n_y*n_x
    
    X = imF.reshape( (n_pixels, n_features) )
    yPred = regressor.predict(X)
    imPred = yPred.reshape( (n_y, n_x) )
    
    predictedImages.append( imPred )
    
  return predictedImages
 
def log( message ):
  print strftime("%H:%M:%S")+' -- '+message
  sys.stdout.flush()

def log_start( message ):
  print strftime("%H:%M:%S")+' -- '+message
  sys.stdout.flush()

def log_done():
  print '...done! '+strftime("%H:%M:%S")
  sys.stdout.flush()

# ==========================================================================
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN 
# ==========================================================================
log('START')
PLOT = False

numTrees = 200
filenamePrefixModel = 'rfRegressorOnAll_noSparse_'+str(numTrees).zfill(2)+'trees_leave'

folderModels = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/models/leaveOneOuts/'
folderPredict = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions/'

# ---------------------------------------------------------------------------------------
# TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING 
# ---------------------------------------------------------------------------------------
folderFeaturesTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/features_ALL/'
folderGtTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/'
trainFeatureFiles = [ os.path.join(folderFeaturesTrain, fn) for fn in os.listdir(folderFeaturesTrain) if fn.endswith('.tif') ]
trainGtFiles = [ os.path.join(folderGtTrain, fn) for fn in os.listdir(folderGtTrain) if fn.endswith('.tif') ]

# -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- 
for i in range(len(trainFeatureFiles)): # leave one out loop
    log_start( 'Start training, leaving out '+str(i+1)+' of '+str(len(trainFeatureFiles))+'... ' )
    regressor, X, y = train(
            trainFeatureFiles[:i]+trainFeatureFiles[(i+1):],
            trainGtFiles[:i]+trainGtFiles[(i+1):],
            subsample_rate=0.25,
            addAllBelowEqual=2,
            numTrees=numTrees )
    log_done()
    log_start( 'Start writing model to "'+folderModels+filenamePrefixModel+str(i)+'"... ' )
    joblib.dump(regressor, folderModels+filenamePrefixModel+str(i)+'.pkl')
    log_done()

    # ---------------------------------------------------------------------------------------
    # TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING
    # ---------------------------------------------------------------------------------------
    testFeatureFiles = [ trainFeatureFiles[i] ]
    log_start( 'Start regressing... ' )
    predImgs = regress( regressor, testFeatureFiles )
    log_done()

    # -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- 
    log_start( 'Start writing results to "'+folderPredict+'"... ' )
    imsave(folderPredict+'prediction'+str(i)+'.tif', predImgs[0].astype(np.float32))
    if PLOT:
        pylab.imshow(predImgs[0], interpolation='nearest')
        pylab.show()
    log_done()
