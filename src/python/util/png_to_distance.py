import numpy as np
from scipy import ndimage as ni
import pylab as pl
import os
from tifffile import imread, imsave


ROOT_DATA = '/Users/abouchar/ownCloud/ProjectRegSeg/data/' # Set to your local path





ROOT = os.path.join(ROOT_DATA,'Histological','BM_dataset_MICCAI2015','annotations_dots')
ROOT_OUT = os.path.join(ROOT_DATA,'Histological','BM_dataset_MICCAI2015','annotations_F_of_distance')
files = [f for f in os.listdir(ROOT) if f.endswith('png')]




for f in files:
    fn_im = os.path.join(ROOT, f)

    im = np.array(pl.mpl.image.imread(fn_im),np.bool)


    imd = ni.morphology.distance_transform_edt(np.invert(im))

    #imd = np.array(np.round(np.maximum(np.exp(alpha*(1-imd/dM))-1,0)),np.int32)
    
    imsave(os.path.join(ROOT_OUT, f[:-4] + '_tfm.tif'),imd,compress=1)
