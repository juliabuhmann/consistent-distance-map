import pylab as pl
import numpy as np
import skimage as ski

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian_filter
from tifffile import *
from skimage.segmentation import find_boundaries

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

fnInputImage = '/Users/jug/ownCloud/ProjectRegSeg/data/Flywing/Medium/Smoothed/t060.tif'
imgInput = imread(fnInputImage)
labels,boundaries,markers,maxima,imgSmoothed = watershed(imgInput, cap=-1)

pl.subplot(1,3,1)
pl.imshow(imgInput)
pl.subplot(1,3,2)
pl.imshow(labels + 10*boundaries)
pl.subplot(1,3,3)
pl.imshow(imgSmoothed + 10*maxima)
# pl.imshow(markers)
# pl.imshow(imgInput-5*boundaries)
pl.show()
