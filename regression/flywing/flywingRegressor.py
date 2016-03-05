# Author: Florian Jug <jug@mpi-cbg.de>

import sys
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
    
    imF = imread(fnF)
    imG = imread(fnG)
    
    n_features = imF.shape[0]
    n_y = imF.shape[1]
    n_x = imF.shape[2]
    n_pixels = n_y*n_x
    
    X_i = imF.reshape( (n_features, n_pixels) ).swapaxes(0,1)
    y_i = imG.reshape( (n_pixels) )

    if 'X' in locals():
      X = np.concatenate((X,X_i))
      y = np.concatenate((y,y_i))
    else:
      X = X_i
      y = y_i
  
  params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
            'learning_rate': 0.01, 'loss': 'ls'}
  clf = ensemble.GradientBoostingRegressor(**params)
  clf.fit(X, y)
  return clf, X, y
  
def regress ( clf, featureStackFilenames ):
  predictedImages = []
  
  for i,fnF in enumerate(featureStackFilenames):
    imF = imread(fnF)
    
    n_features = imF.shape[0]
    n_y = imF.shape[1]
    n_x = imF.shape[2]
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

folderModels = '/Volumes/FastData/ProjectWithJJL/Flywing/'
folderPredict = '/Volumes/FastData/ProjectWithJJL/Flywing/Predictions/'

# ---------------------------------------------------------------------------------------
# TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING 
# ---------------------------------------------------------------------------------------

# -- SMALL or MEDIUM -- SMALL or MEDIUM -- SMALL or MEDIUM -- SMALL or MEDIUM --
# filenameModel = 'regressorOnSmallAll.pkl'
# folderFeaturesTrain = '/Volumes/FastData/ProjectWithJJL/Flywing/Small/FeatureStacks/'
# folderGtTrain = '/Volumes/FastData/ProjectWithJJL/Flywing/Small/GroundTruth/'
# filenameModel = 'regressorOnMediumAll.pkl'
# folderFeaturesTrain = '/Volumes/FastData/ProjectWithJJL/Flywing/Medium/FeatureStacks/'
# folderGtTrain = '/Volumes/FastData/ProjectWithJJL/Flywing/Medium/GroundTruth/'

# trainFeatureFiles = [ folderFeaturesTrain + 'feature-stack0001.tif',
                      # folderFeaturesTrain + 'feature-stack0002.tif',
                      # folderFeaturesTrain + 'feature-stack0003.tif',
                      # folderFeaturesTrain + 'feature-stack0004.tif',
                      # folderFeaturesTrain + 'feature-stack0005.tif',
                      # folderFeaturesTrain + 'feature-stack0006.tif',
                      # folderFeaturesTrain + 'feature-stack0007.tif',
                      # folderFeaturesTrain + 'feature-stack0008.tif',
                      # folderFeaturesTrain + 'feature-stack0009.tif',
                      # folderFeaturesTrain + 'feature-stack0010.tif',
                      # folderFeaturesTrain + 'feature-stack0011.tif' ]
# trainGtFiles = [ folderGtTrain + 't060.tif',
                 # folderGtTrain + 't061.tif',
                 # folderGtTrain + 't062.tif',
                 # folderGtTrain + 't063.tif',
                 # folderGtTrain + 't064.tif',
                 # folderGtTrain + 't065.tif',
                 # folderGtTrain + 't066.tif',
                 # folderGtTrain + 't067.tif',
                 # folderGtTrain + 't068.tif',
                 # folderGtTrain + 't069.tif',
                 # folderGtTrain + 't070.tif' ]

# -- WHOLE WING -- WHOLE WING -- WHOLE WING -- WHOLE WING -- WHOLE WING -- 
filenameModel = 'regressorOnFull_t100.pkl'
folderFeaturesTrain = '/Volumes/FastData/ProjectWithJJL/Flywing/FullWing_t100/FeatureStacks/'
folderGtTrain = '/Volumes/FastData/ProjectWithJJL/Flywing/FullWing_t100/GroundTruth/'
trainFeatureFiles = [ folderFeaturesTrain + 'feature-stack0001.tif' ]
trainGtFiles = [ folderGtTrain + 't100_GT_handCorrection.tif' ]

# -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- TRAIN -- 
log_start( 'Start training... ' )
clf = train(trainFeatureFiles,trainGtFiles)
log_done()
log_start( 'Start writing model to "'+folderModels+filenameModel+'"... ' )
joblib.dump(clf, folderModels+filenameModel)
log_done()

# ---------------------------------------------------------------------------------------
# TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING
# ---------------------------------------------------------------------------------------

# -- SMALL or MEDIUM -- SMALL or MEDIUM -- SMALL or MEDIUM -- SMALL or MEDIUM --
# folderFeaturesTest = '/Volumes/FastData/ProjectWithJJL/Flywing/Small/FeatureStacks/'
# folderGtTest = '/Volumes/FastData/ProjectWithJJL/Flywing/Small/GroundTruth/'
# folderFeaturesTest = '/Volumes/FastData/ProjectWithJJL/Flywing/Medium/FeatureStacks/'
# folderGtTest = '/Volumes/FastData/ProjectWithJJL/Flywing/Medium/GroundTruth/'

# testFeatureFiles = [ folderFeaturesTest + 'feature-stack0001.tif',
                     # folderFeaturesTest + 'feature-stack0002.tif',
                     # folderFeaturesTest + 'feature-stack0003.tif',
                     # folderFeaturesTest + 'feature-stack0004.tif',
                     # folderFeaturesTest + 'feature-stack0005.tif',
                     # folderFeaturesTest + 'feature-stack0006.tif',
                     # folderFeaturesTest + 'feature-stack0007.tif',
                     # folderFeaturesTest + 'feature-stack0008.tif',
                     # folderFeaturesTest + 'feature-stack0009.tif',
                     # folderFeaturesTest + 'feature-stack0010.tif',
                     # folderFeaturesTest + 'feature-stack0011.tif' ]
# testGtFiles = [ folderGtTest + 't060.tif',
                # folderGtTest + 't061.tif',
                # folderGtTest + 't062.tif',
                # folderGtTest + 't063.tif',
                # folderGtTest + 't064.tif',
                # folderGtTest + 't065.tif',
                # folderGtTest + 't066.tif',
                # folderGtTest + 't067.tif',
                # folderGtTest + 't068.tif',
                # folderGtTest + 't069.tif',
                # folderGtTest + 't070.tif' ]

# -- WHOLE WING -- WHOLE WING -- WHOLE WING -- WHOLE WING -- WHOLE WING -- 
folderFeaturesTest = '/Volumes/FastData/ProjectWithJJL/Flywing/FullWing_t100/FeatureStacks/'
folderGtTest = '/Volumes/FastData/ProjectWithJJL/Flywing/FullWing_t100/GroundTruth/'
testFeatureFiles = [ folderFeaturesTest + 'feature-stack0001.tif' ]

# -- LOAD MODEL -- LOAD MODEL -- LOAD MODEL -- LOAD MODEL -- LOAD MODEL -- 
# filenameModel = 'regressorOnSmallAll.pkl'
filenameModel = 'regressorOnMediumAll.pkl'
# filenameModel = 'regressorOnFull_t100.pkl'
log_start( 'Start reading model from "'+folderModels+filenameModel+'"... ' )
clf = joblib.load( folderModels+filenameModel )
log_done()
log_start( 'Start regressing... ' )
predImgs = regress( clf[0], testFeatureFiles )
log_done()

# -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- WRITE RESULTS -- 
log_start( 'Start writing results to "'+folderPredict+'"... ' )
for i,img in enumerate(predImgs):
  imsave(folderPredict+'prediction'+str(i)+'.tif', img)
  pylab.imshow(img, interpolation='nearest')
  pylab.show()
log_done()
