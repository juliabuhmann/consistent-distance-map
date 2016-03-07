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

def train( featureStackFilenames, groundTruthFilenames ):
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

    if 'X' in locals():
      X = np.concatenate((X,X_i))
      y = np.concatenate((y,y_i))
    else:
      X = X_i
      y = y_i

  params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
            'learning_rate': 0.01, 'loss': 'ls', 'verbose': 3}
  clf = ensemble.GradientBoostingRegressor(**params)
  clf.fit(X, y)
  return clf, X, y

def regress ( clf, featureStackFilenames ):
  predictedImages = []

  for i,fnF in enumerate(featureStackFilenames):
    print ' >> Starting to regress image ', i, ' of ', len(featureStackFilenames)

    imF = np.swapaxes(np.swapaxes(imread(fnF),0,2),0,1) 
    
    n_features = imF.shape[3]*imF.shape[2]
    n_y = imF.shape[0]
    n_x = imF.shape[1]
    n_pixels = n_y*n_x
    
    X = imF.reshape( (n_pixels, n_features) )
    yPred = clf.predict(X)
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

filenameModel = 'regressorOnAll.pkl'

folderModels = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/models/'
folderPredict = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/predictions/'

# ---------------------------------------------------------------------------------------
# TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING 
# ---------------------------------------------------------------------------------------
folderFeaturesTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/features/'
folderGtTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance/'
trainFeatureFiles = [ os.path.join(folderFeaturesTrain, fn) for fn in os.listdir(folderFeaturesTrain) if fn.endswith('.tif') ]
trainGtFiles = [ os.path.join(folderGtTrain, fn) for fn in os.listdir(folderGtTrain) if fn.endswith('.tif') ]

# print trainFeatureFiles
# print trainGtFiles

# -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- 
# log_start( 'Start training... ' )
# clf = train(trainFeatureFiles,trainGtFiles)
# log_done()
# log_start( 'Start writing model to "'+folderModels+filenameModel+'"... ' )
# joblib.dump(clf, folderModels+filenameModel)
# log_done()

# ---------------------------------------------------------------------------------------
# TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING
# ---------------------------------------------------------------------------------------
folderFeaturesTest = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/features_test/'
folderGtTest = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/ICPR_Lymphocytes/ground_truth_distance_test/'

testFeatureFiles = [ os.path.join(folderFeaturesTest, fn) for fn in os.listdir(folderFeaturesTest) if fn.endswith('.tif') ]
testGtFiles = [ os.path.join(folderGtTest, fn) for fn in os.listdir(folderGtTest) if fn.endswith('.tif') ]

# -- LOAD MODEL -- LOAD MODEL -- LOAD MODEL -- LOAD MODEL -- LOAD MODEL -- 
log_start( 'Start reading model from "'+folderModels+filenameModel+'"... ' )
clf = joblib.load( folderModels+filenameModel )
log_done()
log_start( 'Start regressing... ' )
predImgs = regress( clf[0], testFeatureFiles )
log_done()

# -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- 
log_start( 'Start writing results to "'+folderPredict+'"... ' )
for i,img in enumerate(predImgs):
  imsave(folderPredict+'prediction'+str(i)+'.tif', img.astype(np.float32))
  if PLOT:
      pylab.imshow(img, interpolation='nearest')
      pylab.show()
log_done()
