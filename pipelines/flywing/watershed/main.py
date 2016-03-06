import pylab as pl
import numpy as np
import skimage as ski

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian_filter
from tifffile import *
from skimage.segmentation import find_boundaries

PLOT = False
SAVE = True

def watershed(imgInput, cap=-1, sigma=1.0, min_dist_maxima=4):
    normFac = imgInput.max()
    imgSmoothed = normFac * gaussian_filter(imgInput/normFac, sigma=sigma)
    if cap==-1:
        cap = imgInput.max()
    imgCapped = np.minimum(imgSmoothed, cap*np.ones_like(imgSmoothed))
    # imgCapped = np.minimum(imgInput, cap*np.ones_like(imgInput))
    # imgSmoothed = gaussian_filter(imgCapped/imgCapped.max(), sigma=sigma)
    local_maxi = peak_local_max(imgCapped, indices=False, exclude_border=False, min_distance=min_dist_maxima )
    markers = ndi.label(local_maxi)[0]
    labels = ski.morphology.watershed(-imgInput, markers)
    boundaries = find_boundaries(labels, mode='thick')
    return labels, boundaries, markers, local_maxi, imgCapped

def run( fnInputImagePrefix, fnOutputImagePrefix, indexRange ):
    for i in indexRange:
        imgInput = imread(fnInputImagePrefix+str(i)+'.tif')
        labels,boundaries,markers,maxima,imgSmoothed = watershed(imgInput, cap=-1)

        if SAVE:
            imsave(fnOutputImagePrefix+'labels_t0'+str(i)+'.tif', labels)
            imsave(fnOutputImagePrefix+'boundaries_t0'+str(i)+'.tif', boundaries.astype(np.int8))
            imsave(fnOutputImagePrefix+'maxima_t0'+str(i)+'.tif', maxima.astype(np.int8))
            imsave(fnOutputImagePrefix+'smoothed_t0'+str(i)+'.tif', imgSmoothed.astype(np.float32))

        if PLOT:
            pl.subplot(1,3,1)
            pl.imshow(imgInput)
            pl.subplot(1,3,2)
            pl.imshow(labels + 10*boundaries)
            pl.subplot(1,3,3)
            pl.imshow(imgSmoothed + 10*maxima)
 
print 'Starting run 1...'
fnInputImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Smoothed/t0'
fnOutputImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Smoothed/seg_'
run( fnInputImagePrefix, fnOutputImagePrefix, range(60,71) )

print 'Starting run 2...'
fnInputImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Predictions/prediction'
fnOutputImagePrefix = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Predictions/seg_'
run( fnInputImagePrefix, fnOutputImagePrefix, range(0,11) )
