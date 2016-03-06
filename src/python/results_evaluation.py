'''
Module containing all methods to evaluate the quality of the different methods.


# Methods
# -------


## score:
    
    # arguments:
        - image: numpy.ndarray
            Image whose quality is assessed.
        - ground_truth: numpy.ndarray
            Reference to which image is compared.
        - score: string
            Measure to compute for comparison. So far, 'L1', 'L2', 'VI',
            'CC_VI' and 'percentage' are valid inputs.


'''


### Imports
import numpy as np
#import pymaxflow
from scipy import ndimage as ni
import h5py
import os
from time import time
import sys
import resource
from subprocess import call, STDOUT
import validate_distance_maps as vdm




def score(image, ground_truth, score='L1'):
    '''
    # Parameters
    ============
    
    image: numpy.ndarray
        Array containing the regressed distance.
    
    ground_truth: numpy.ndarray
        Array containing the true distance.
        
    score: string
        One of 'L1', 'L2', 'VI', 'CC_VI'.
        
    '''
    
    assert all(np.array(image.shape) == ground_truth.shape)
    
    if str.upper(score) == 'L1':
        
        return np.mean(np.abs(image-ground_truth))
        
    elif str.upper(score) == 'L2':
        
        return np.mean((image-ground_truth)**2)
        
    elif str.upper(score) == 'VI':
        
        score_ = 0
        n = float(np.size(image))
        
        for val in np.unique(image):
            for val2 in np.unique(ground_truth):
                idx = image==val
                idy = ground_truth==val2
                
                nx = np.sum(idx)/n
                ny = np.sum(idy)/n
                
                rxy = np.sum(np.logical_and(idx,idy))/n
                
                if rxy > 0:
                    score_ -= rxy*(np.log(rxy/nx)+np.log(rxy/ny))
            
        return score_
        
    elif str.upper(score) == 'CC_VI':
        im1 = _get_segmentation(np.invert(image.astype(np.bool)))[0]
        im2 = _get_segmentation(np.invert(ground_truth.astype(np.bool)))[0]
        
        return _varinfo(im1,im2)
    
    elif str.lower(score) in ['percentage', 'perc', 'p']:
        
        return vdm.calculate_perc_of_correct(ground_truth.astype(np.int), image.astype(np.int))*100    
    else:
        
        raise Exception("Not recognized")

def _get_segmentation(binary_image):
    
    return ni.label(binary_image)



def _conn_comp_VI(binary_image1, binary_image2,edge_image=True):
    # Replaced by _varinfo
    
    
    
    if edge_image:
        seg1, n1 = _get_segmentation(np.invert(binary_image1))
        seg2, n2 = _get_segmentation(np.invert(binary_image2))
    else:
        seg1, n1 = _get_segmentation(binary_image1)
        seg2, n2 = _get_segmentation(binary_image2)
    
    score_ = 0.
    n = np.size(binary_image1)
    for val in range(1,n1+1):
        for val2 in range(1,n2+1):
            id1 = seg1==val
            id2 = seg2==val2
            
            nx = np.sum(id1)*1.
            ny = np.sum(id2)*1.
            
            #rxy = np.sum(np.logical_and(id1,id2))*1./n
            rxyn = np.sum(np.logical_and(id1,id2))*1.
            
            if rxyn > 0.:
                score_ -= rxyn*(np.log(rxyn/nx)+np.log(rxyn/ny))/n
            #if stop:
                #print 'n: %d, nx: %f, ny: %f, rxy: %f, score_diff: %f' % (n,nx,ny,rxy,rxy*(np.log(rxy/nx)+np.log(rxy/ny)))
    
                    
    return score_    
  
  
def _entropy(label):
    N = np.size(label)
    #~ N = np.sum(label>0)
    k = [el for el in np.unique(label)]# if el != 0]
    #~ k = [el for el in np.unique(label) if el != 0]
    H = 0.
      
      
    for i in k:
        pk = float(np.sum(i == label))/N
        H -= pk*np.log(pk)
        if np.isnan(H):
            raise Exception()
    return H
    
def _varinfo(label1,label2):
    
    h1 = _entropy(label1)
    h2 = _entropy(label2)

    i12 = _mutualinfo(label1,label2)
    
    return h1 + h2 - 2*i12

def _mutualinfo(label1,label2): 
    
    N = float(np.size(label1))
    #~ N = float(np.sum(label1+label2>0))
    k1 = [el for el in np.unique(label1)]# if el != 0]
    k2 = [el for el in np.unique(label2)]# if el != 0]
    #~ k1 = [el for el in np.unique(label1) if el != 0]
    #~ k2 = [el for el in np.unique(label2) if el != 0]
    I = 0


    for i in k1:
        # loop over the unique elements of L2
        for j in k2:
            # the mutual probability of two classification indices occurring in
            # L1 and L2
            pij = np.sum((label1 == i)*(label2 == j))/N
            # the probability of a given classification index occurring in L1
            pi = np.sum(label1 == i)/N
            # the probability of a given classification index occurring in L2
            pj = np.sum(label2 == j)/N
            if pij > 0:
                I += pij*np.log(pij/(pi*pj))
            
    return I
        
def best_thresh(image, ground_truth, max_dist=None, score_func='L1'):
    
    if max_dist == None:
        max_dist = int(np.maximum(  np.max(image), np.max(ground_truth) ) )
    
    thresholds = range(1,max_dist-1)
    
    scores = [score(image<threshold,ground_truth<1,score_func) for threshold in thresholds]
    
    if score_func in ['percentage', 'perc', 'p']:
        return np.max(scores), thresholds[np.argmax(scores)]
    else:
        return np.min(scores), thresholds[np.argmin(scores)]


def plot_histogram(values, ground_truth, max_dist):
    import matplotlib.pyplot as plt
    plt.hist2d(ground_truth, values, (max_dist+1, max_dist+1), cmap=plt.cm.jet, norm=mpl.colors.LogNorm())
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.title('2d histogram of distance to membrane results')
    plt.show()

def print_scores(ground_truth, noisy_distance, smoothed_distance):
    
    #--------------------------------------------------------------------------
    # Noisy distance scores
    #--------------------------------------------------------------------------
    score_pred, T_pred = best_thresh(noisy_distance, ground_truth, score_func='L1')
    VI_pred, T_pred_VI = best_thresh(noisy_distance, ground_truth, score_func='VI')
    CC_VI_pred, T_pred_CC_VI = best_thresh(noisy_distance, ground_truth, score_func='CC_VI')
    perc_pred, T_pred_perc = best_thresh(noisy_distance, ground_truth, score_func='percentage')
    
    VI_pred_dist = score(noisy_distance, ground_truth,'VI')
    L1_err_pred = score(noisy_distance, ground_truth,'L1')
    L2_err_pred = score(noisy_distance, ground_truth,'L2')
    perc_pred = score(noisy_distance, ground_truth,'percentage')
    
    
    

    print "\n\n\t\t----------------------"
    print '\033[1m' + "\t\t\tSCORES" + '\033[0m'
    print "\t\t----------------------\n"
    print "CNN prediction:\n"
    print "\t-Segmentation:\n"
    print "\t    -L1:"
    print "\t        Best threshold: %d" % T_pred
    print "\t        Error: %.5f" % score_pred
    print "\t    -Percentage correct:"
    print "\t        Best threshold: %d" % T_pred_perc
    print "\t        %% correct: %.5f" % perc_pred
    print "\t    -Variation of Information:"
    print "\t        Best threshold: %d"% T_pred_VI
    print "\t        Error: %.5f" % VI_pred
    print "\t    -Variation of Information on Connected Components:"
    print "\t        Best threshold: %d"% T_pred_CC_VI
    print "\t        Error: %.5f\n" % CC_VI_pred
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_pred
    print "\t    -L2 error: %.1f" % L2_err_pred
    print "\t    -Percentage correct: %.2f%%" % perc_pred
    print "\t    -VI score: %.3f\n\n" % VI_pred_dist    
    
    
    
    
    score_smoothed, T_smoothed = best_thresh(smoothed_distance, ground_truth, score_func='L1')
    VI_smoothed, T_smoothed_VI = best_thresh(smoothed_distance, ground_truth, score_func='VI')
    CC_VI_smoothed, T_smoothed_CC_VI = best_thresh(smoothed_distance, ground_truth, score_func='CC_VI')
    perc_smoothed, T_smoothed_perc = best_thresh(smoothed_distance, ground_truth, score_func='percentage')
    
    VI_smoothed_dist = score(smoothed_distance, ground_truth,'VI')
    L1_err_smoothed = score(smoothed_distance, ground_truth,'L1')
    L2_err_smoothed = score(smoothed_distance, ground_truth,'L2')
    perc_smoothed = score(smoothed_distance, ground_truth,'percentage')
  
  
    print "Smoothed prediction:\n"
    print "\t-Segmentation:\n"
    print "\t    -L1:"
    print "\t        Best threshold: %d" % T_smoothed
    print "\t        Error: %.5f" % score_smoothed
    print "\t    -Percentage correct:"
    print "\t        Best threshold: %d" % T_smoothed_perc
    print "\t        %% correct: %.5f" % perc_smoothed
    print "\t    -Variation of Information:"
    print "\t        Best threshold: %d"% T_smoothed_VI
    print "\t        Error: %.5f" % VI_smoothed
    print "\t    -Variation of Information on Connected Components:"
    print "\t        Best threshold: %d"% T_smoothed_CC_VI
    print "\t        Error: %.5f\n" % CC_VI_smoothed
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_err_smoothed
    print "\t    -L2 error: %.1f" % L2_err_smoothed
    print "\t    -Percentage correct: %.2f%%" % perc_smoothed
    print "\t    -VI score: %.3f\n" % VI_smoothed_dist 
