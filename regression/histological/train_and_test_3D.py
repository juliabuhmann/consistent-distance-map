import os
import numpy as np
from tifffile import imread
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pylab as pl
import sys
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(ROOT_PATH,'src','python'))
import surface_reconstruction as sr

n_samples_training = 1000000

alpha = 5.
dM = 39.

ROOT_DATA = '/Users/abouchar/ownCloud/ProjectRegSeg/data/' # Set to your local path




### 0. Set up a few variables

ROOT = os.path.join(ROOT_DATA,'Flybrain','GroundTruthCenterline')
ROOT_FTR = os.path.join(ROOT_DATA,'Flybrain','features')
ROOT_IMG = os.path.join(ROOT_DATA,'Flybrain','RawData')
ROOT_DST = os.path.join(ROOT_DATA,'Flybrain','GroundTruth')
files = [f for f in os.listdir(ROOT) if f.endswith('tif')]


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

prog = os.path.join(ROOT_PATH,'..','src','cpp','graph_cut')

# list of files
files_ftr = [f for f in os.listdir(ROOT_FTR) if f.endswith('tif')]
files_dst = [f for f in os.listdir(ROOT_DST) if f.endswith('tif')]
files_img = [f for f in os.listdir(ROOT_IMG) if f.endswith('tif')]




### 1. Run training and testing
img_size_X = 1200
img_size_Y = 1200
N_channels = 3
N_features = 79


train_image = 0
test_image = 1

test_square = [300,300, 256, 256] # upper-left and size_x, size_y

fn_ftr,fn_dst = (files_ftr[train_image],files_dst[train_image])

train_features = np.swapaxes(np.swapaxes(imread(os.path.join(ROOT_FTR,fn_ftr)),0,2),0,1) # Images have size 3 x size_X x size_Y x N_features_per_channel. Reshape to size_X x size_Y x 3 x N_features_per_channel
train_distances = imread(os.path.join(ROOT_DST,fn_dst))

X = train_features.reshape(img_size_X*img_size_Y,N_channels*N_features)
y = train_distances.flatten()

clf = GradientBoostingRegressor()

clf.fit(X[:n_samples_training,:],y[:n_samples_training])

fn_ftr,fn_dst,fn_img = (files_ftr[1],files_dst[1],files_img[1])


start_x = test_square[0]
start_y = test_square[1]
end_x = start_x + test_square[2]
end_y = start_y + test_square[3]

test_features = np.swapaxes(np.swapaxes(imread(os.path.join(ROOT_FTR,fn_ftr)),0,2),0,1)[start_x:end_x,start_y:end_y,:,:]
test_distances = imread(os.path.join(ROOT_DST,fn_dst))[start_x:end_x,start_y:end_y]
img = pl.mpl.image.imread(os.path.join(ROOT_IMG,fn_img))[start_x:end_x,start_y:end_y,:]

X = test_features.reshape(img_size_X*img_size_Y,N_features*N_channels)
#~ y = test_distances.flatten()

y_pred = clf.predict(X)

cmap = pl.get_cmap('YlGnBu')
cmap.set_under([0.7,0.95,0.5])
cmap.set_under([1,0,0.])
pl.figure()
pl.subplot(2,3,1)
pl.title('Original Image')
pl.imshow(img,interpolation='nearest')
pl.subplot(2,3,2)
pl.title('Ground Truth')
pl.imshow(test_distances,interpolation='nearest')
pl.subplot(2,3,3)
pl.title('Prediction')
pl.imshow(np.round(np.maximum(y_pred.reshape(size_test,size_test),0)),interpolation='nearest',cmap=cmap)



pl.subplot(2,3,4)

d = dM*( 1.  -  (np.log(np.maximum(y_pred.reshape(size_test,size_test),0)+1) / alpha))
#d[np.isnan(d)] = 0.
pl.title('Prediction converted to Distance')
pl.imshow(np.round(d),interpolation='nearest',cmap=cmap)

VERBOSE=True
pl.subplot(2,3,5)
max_dist=16
D = np.minimum(np.round(d),max_dist)

results = sr.reconstruct_surface(D,'/Users/abouchar/Desktop/dump/temp.txt','/Users/abouchar/Desktop/dump/temp_out.txt',prog,overwrite=True, max_dist=max_dist, sampling = [1,1], cost_fun='linear', verbose=VERBOSE)

#d[np.isnan(d)] = 0.
pl.imshow(results,interpolation='nearest',cmap=cmap)
pl.title('Smoothed Distance')

pl.subplot(2,3,6)
d = np.array(np.round(np.maximum(np.exp(alpha*(1-results.reshape(size_test,size_test).astype(np.float)/dM))-1,0)),np.int32)
pl.imshow(d,interpolation='nearest',cmap=cmap)
pl.title('Converted Smoothed Distance')
pl.show()
