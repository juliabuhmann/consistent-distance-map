import numpy as np
from scipy import ndimage as ni

'''
Computes distance transform of a given (boolean) image (numpy array).
'''
def distTransform( img, max_dist=-1, fg_is_one=True, do_lepetit_dist=False, dM=39, alpha=5 ):
    if fg_is_one:
        imgDist = ni.morphology.distance_transform_edt(np.invert(img))
    else:
        imgDist = ni.morphology.distance_transform_edt(img)

    if do_lepetit_dist:
        imgDist = np.array(np.round(np.maximum(np.exp(alpha*(1-imgDist/dM))-1,0)),np.int32)

    if (max_dist>=0):
        imgDist = np.minimum(imgDist, max_dist*np.ones_like(imgDist))

    return imgDist
