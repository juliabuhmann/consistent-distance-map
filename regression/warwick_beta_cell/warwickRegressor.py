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

    print y_i.size

    # filter out useless training data
    max_dist = y_i.max()
    elements_to_take = y_i<max_dist
    elements_to_take = np.logical_or( elements_to_take, np.random.rand(y_i.size)<0.01 )
    X_i = X_i[elements_to_take]
    y_i = y_i[elements_to_take]

    print y_i.size

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
    imF = np.swapaxes(np.swapaxes(imread(fnF),0,2),0,1) 
    
    n_features = imF.shape[3]*imF.shape[2]
    n_y = imF.shape[0]
    n_x = imF.shape[1]
    n_pixels = n_y*n_x
    
    X = imF.reshape( (n_features, n_pixels) ).swapaxes(0,1)
    yPred = clf.predict(X);
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

filenameModel = 'regressorOnLuxian.pkl'

folderModels = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/models/'
folderPredict = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/predictions/'

# ---------------------------------------------------------------------------------------
# TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING 
# ---------------------------------------------------------------------------------------
folderFeaturesTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/features/'
folderGtTrain = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/ground_truth_distmaps/'
trainFeatureFiles = [ os.path.join(folderFeaturesTrain, fn) for fn in os.listdir(folderFeaturesTrain) if fn.endswith('.tif') ]
trainGtFiles = [ os.path.join(folderGtTrain, fn) for fn in os.listdir(folderGtTrain) if fn.endswith('.tif') ]

# print trainFeatureFiles
# print trainGtFiles

# -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- 
log_start( 'Start training... ' )
clf = train(trainFeatureFiles[0:2],trainGtFiles[0:2])
log_done()
log_start( 'Start writing model to "'+folderModels+filenameModel+'"... ' )
joblib.dump(clf, folderModels+filenameModel)
log_done()

# ---------------------------------------------------------------------------------------
# TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING
# ---------------------------------------------------------------------------------------
folderFeaturesTest = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/features_test/'
folderGtTest = '/Users/jug/ownCloud/ProjectRegSeg/data/Histological/warwick_beta_cell/ground_truth_distmaps_test/'

trainFeatureFiles = [ os.path.join(folderFeaturesTest, fn) for fn in os.listdir(folderFeaturesTest) if fn.endswith('.tif') ]
trainGtFiles = [ os.path.join(folderGtTest, fn) for fn in os.listdir(folderGtTest) if fn.endswith('.tif') ]

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
