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

def train( featureStackFilenames, groundTruthFilenames, doClassEqualization=False ):
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

    # add 0 and 1 distances more then only one time
    if doClassEqualization:
        print y_i.size,
        X_i_0 = X_i[y_i==0]
        y_i_0 = y_i[y_i==0]
        X_i_01 = X_i[y_i<=1]
        y_i_01 = y_i[y_i<=1]
        # raise Exception
        X_i=np.concatenate( (X_i,np.concatenate( (X_i_01, X_i_0) ,axis=0)) ,axis=0)
        y_i=np.concatenate( (y_i,np.concatenate( (y_i_01, y_i_0) ,axis=0)) ,axis=0)

    print y_i.size,

    # filter out useless training data
    # max_dist = y_i.max()
    # elements_to_take = y_i<max_dist
    # elements_to_take = np.logical_or( elements_to_take, np.random.rand(y_i.size)<0.01 )
    # X_i = X_i[elements_to_take]
    # y_i = y_i[elements_to_take]

    print y_i.size

    if 'X' in locals():
      X = np.concatenate((X,X_i))
      y = np.concatenate((y,y_i))
    else:
      X = X_i
      y = y_i

  # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
            # 'learning_rate': 0.01, 'loss': 'lad', 'verbose': 3}
  # clf = ensemble.GradientBoostingRegressor(**params)
  # clf.fit(X, y)
  # return clf, X, y

  params = {'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2,
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
PLOT = True

filenamePrefixModel = 'rfRegressorOnAll_noSparse_100trees_leave'

folderModels = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/models/leaveOneOuts/'
folderPredict = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions/'

# ---------------------------------------------------------------------------------------
# TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING 
# ---------------------------------------------------------------------------------------
folderFeaturesTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/features_ALL/'
folderGtTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/'
trainFeatureFiles = [ os.path.join(folderFeaturesTrain, fn) for fn in os.listdir(folderFeaturesTrain) if fn.endswith('.tif') ]
trainGtFiles = [ os.path.join(folderGtTrain, fn) for fn in os.listdir(folderGtTrain) if fn.endswith('.tif') ]

# print trainFeatureFiles
# print trainGtFiles

# -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- 
for i in range(len(trainFeatureFiles)): # leave one out loop
    log_start( 'Start training, leaving out '+str(i+1)+' of '+str(len(trainFeatureFiles))+'... ' )
    regressor = train(
            trainFeatureFiles[:i]+trainFeatureFiles[(i+1):],
            trainGtFiles[:i]+trainGtFiles[(i+1):],
            doClassEqualization=True)
    log_done()
    log_start( 'Start writing model to "'+folderModels+filenamePrefixModel+str(i)+'"... ' )
    joblib.dump(regressor, folderModels+filenamePrefixModel+str(i)+'.pkl')
    log_done()

    # ---------------------------------------------------------------------------------------
    # TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING
    # ---------------------------------------------------------------------------------------
    testFeatureFiles = [ trainFeatureFiles[i] ]
    log_start( 'Start regressing... ' )
    predImgs = regress( regressor[0], testFeatureFiles )
    log_done()

    # -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- 
    log_start( 'Start writing results to "'+folderPredict+'"... ' )
    imsave(folderPredict+'prediction'+str(i)+'.tif', predImgs[0].astype(np.float32))
    if PLOT:
        pylab.imshow(predImgs[0], interpolation='nearest')
        pylab.show()
    log_done()
