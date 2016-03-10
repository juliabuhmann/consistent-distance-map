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
import validate_distance_maps as vdm


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def score(image, ground_truth, score='L1'):
    '''
    # Parameters
    ============
    
    image: numpy.ndarray
        Array containing the regressed distance.
    
    ground_truth: numpy.ndarray
        Array containing the true distance.
        
    score: string
        One of 'L1', 'L2', 'VI', 'CC_VI','TP','FP','TN','FN'.
        
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
    
    elif str.upper(score) == 'TP':
        
        return np.sum(np.logical_and(ground_truth==0,image==0)) 
           
    
    elif str.upper(score) == 'FP':
        
        return np.sum(np.logical_and(ground_truth!=0,image==0))
           
    
    elif str.upper(score) == 'TN':
        
        return np.sum(np.logical_and(ground_truth!=0,image!=0)) 
           
    
    elif str.upper(score) == 'FN':
        
        return np.sum(np.logical_and(ground_truth==0,image!=0)) 
    
    elif str.upper(score) in ['P','PREC','PRECISION']:
        
        return float(np.sum(np.logical_and(ground_truth==0,image==0)))/np.sum(image==0)
        
    elif str.upper(score) in ['R','REC','RECALL', 'TPR']: # Recall or True Positive Rate
        
        return float(np.sum(np.logical_and(ground_truth==0,image==0)))/np.sum(ground_truth==0)
        
    elif str.upper(score) in ['FPR']: # False Positive Rate
        
        return float(np.sum(np.logical_and(ground_truth!=0,image==0)))/np.sum(ground_truth!=0)
        
    elif str.upper(score) in ['S','SPEC','SPECIFICITY','TNR']: # Specificity or True Negative Rate
        
        return float(np.sum(np.logical_and(ground_truth!=0,image!=0)))/np.sum(ground_truth!=0)
        
    elif str.upper(score) in ['FNR']: # False Negative Rate
        
        return float(np.sum(np.logical_and(ground_truth==0,image!=0)))/np.sum(ground_truth==0)
        
    elif str.upper(score) in ['A', 'ACCURACY']: # Accuracy
        
        return float(np.sum(ground_truth==image))/np.size(ground_truth)
           
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

def compare_scores(ground_truth, predictions, titles, seg_scores=None, dist_scores=None, format_best_score='Green'):
    
    if isinstance(titles, str):
        
        assert np.shape(np.squeeze(ground_truth)) == np.shape(np.squeeze(predictions))
        
        titles = [titles]
        predictions = [predictions]
        
    elif isinstance(titles, list):
        
        for prediction in predictions:
            
            assert np.shape(np.squeeze(ground_truth)) == np.shape(np.squeeze(prediction))
    
    
    assert len(predictions) == len(titles)
    
    
    
    if isinstance(format_best_score,str):
        if format_best_score.lower() == 'green':
            fmt_best = bcolors.OKGREEN
        elif format_best_score.lower() == 'blue':
            fmt_best = bcolors.OKBLUE
        elif format_best_score.lower() == 'red':
            fmt_best = bcolors.FAIL
        elif format_best_score.lower() == 'yellow':
            fmt_best = bcolors.WARNING
        elif format_best_score.lower() == 'bold':
            fmt_best = bcolors.BOLD
        elif format_best_score.lower() == 'underline':
            fmt_best = bcolors.UNDERLINE
    
    
    if seg_scores is None and dist_scores is None:
        
        seg_scores = ['L1','CC_VI','perc','precision','recall']
        
        dist_scores = ['L1','L2','VI','perc']
    
    score_names = {'L1':'L1 Error',
                   'L2':'L2 Error',
                   'CC_VI':'Var of Info on CC',
                   'VI':'Variation of Info',
                   'perc':'% correct values',
                   'precision':'Precision',
                   'recall':'Recall'}
                   
    best_score =  {'L1':np.argmin,
                   'L2':np.argmin,
                   'CC_VI':np.argmin,
                   'VI':np.argmin,
                   'perc':np.argmax,
                   'precision':np.argmax,
                   'recall':np.argmax}
    
    print "\n\n"
    print "".join([' ']*int((WIDTH-22)*0.5)) + "######################"
    print "".join([' ']*int((WIDTH-22)*0.5)) + '\033[1m' + "#       SCORES       #" + '\033[0m'
    print "".join([' ']*int((WIDTH-22)*0.5)) + "######################\n"
    
    
    if len(titles) > 3: # Won't fit in a table
        for prediction, title in zip(predictions, titles):
            print_scores(ground_truth, prediction, title=title)
            print "\n"
    
    else:
        width = np.maximum(WIDTH_SCORE,int(WIDTH*1./(len(titles)+1)))
        
        print_title_bar(titles, width=width, flag=bcolors.BOLD)
        
        print_hbar('-')
        print_title_bar(['Threshold, Score']*len(predictions),'Segmentation', width=width, flag=bcolors.WARNING)
        for score_ in seg_scores:
            scores = []
            thresholds = []
            for prediction in predictions:
                S,T = best_thresh(prediction, ground_truth, score_func=score_)
                scores.append(S)
                thresholds.append(T)
            
            values = ['%d, %.4f' % (t,s) for t,s in zip(thresholds,scores)]
            
            print_title_bar(values,score_names[score_], width=width, bold_ind=best_score[score_](scores), fmt_best=fmt_best)    
            
            
                
                
        print_hbar('-')
        print_title_bar(['Score']*len(predictions),'Distances', width=width, flag=bcolors.WARNING)
        for score_ in dist_scores:
            scores = []
            thresholds = []
            for prediction in predictions:
                scores.append(score(prediction,ground_truth,score_))
            
            values = ['%.4f' % s for s in scores]
            
            print_title_bar(values,score_names[score_], width=width, bold_ind=best_score[score_](scores), fmt_best=fmt_best)      
        
        
    
    #~ for prediction, title in zip(predictions, titles):
        #~ print_scores(ground_truth, prediction, title=title)
        #~ print "\n"

WIDTH = 80
WIDTH_SCORE = 20

def print_title_bar(titles,first_col=' ',width=WIDTH_SCORE, flag=None, fmt_first = False, bold_ind=None, fmt_best=bcolors.OKGREEN):
    
    if flag is None or fmt_first:
        fmt_titles = [title if len(title) <= width-2 else title[:width-2] for title in titles]
    else:
        fmt_titles = [flag + title + bcolors.ENDC if len(title) <= width-2 else title[:width-2] for title in titles]
    
    ls = [width-len(title) if flag is None else width-len(title)+len(flag)+len(bcolors.ENDC) for title in fmt_titles]
    
    fmt_title = [title.ljust(len(title)+int(0.5*l)) for title,l in zip(fmt_titles,ls)]
    
    fmt_title = [title.rjust(width) if flag is None else title.rjust(width+len(flag)+len(bcolors.ENDC)) for title in fmt_titles]
    
    if not bold_ind is None:
        fmt_title = [f if i != bold_ind else fmt_best + f + bcolors.ENDC for i,f in enumerate(fmt_title)]
    
    if flag is None:
        print first_col + ' '.ljust(width-len(first_col)) +  ''.join(fmt_title)
    else:
        print flag + first_col + bcolors.ENDC + ' '.ljust(width-len(first_col)) +  ''.join(fmt_title)
    
    
def print_hbar(char='-',width=WIDTH):
    
    print ''.join([char]*width)
    
    
def print_scores(ground_truth, estimate, title='Prediction'):
    
    #--------------------------------------------------------------------------
    # Noisy distance scores
    #--------------------------------------------------------------------------
    best_L1, T_L1 = best_thresh(estimate, ground_truth, score_func='L1')
    best_VI, T_VI = best_thresh(estimate, ground_truth, score_func='VI')
    best_CC_VI, T_CC_VI = best_thresh(estimate, ground_truth, score_func='CC_VI')
    best_perc, T_perc = best_thresh(estimate, ground_truth, score_func='percentage')
    
    VI_dist = score(estimate, ground_truth,'VI')
    L1_dist = score(estimate, ground_truth,'L1')
    L2_dist = score(estimate, ground_truth,'L2')
    perc_dist = score(estimate, ground_truth,'percentage')
    
    
    
    print title + ":\n"
    print "\t-Segmentation:\n"
    print "\t    -L1:"
    print "\t        Best threshold: %d" % T_L1
    print "\t        Error: %.5f" % best_L1
    print "\t    -Percentage correct:"
    print "\t        Best threshold: %d" % T_perc
    print "\t        %% correct: %.5f" % best_perc
    print "\t    -Variation of Information:"
    print "\t        Best threshold: %d"% T_VI
    print "\t        Error: %.5f" % best_VI
    print "\t    -Variation of Information on Connected Components:"
    print "\t        Best threshold: %d"% T_CC_VI
    print "\t        Error: %.5f\n" % best_CC_VI
    print "\t-Distance map:\n"
    print "\t    -L1 error: %.3f" % L1_dist
    print "\t    -L2 error: %.1f" % L2_dist
    print "\t    -Percentage correct: %.2f%%" % perc_dist
    print "\t    -VI score: %.3f\n" % VI_dist    
    
    
    
