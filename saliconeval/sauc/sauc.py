#!/usr/bin/env python
# 
# File Name : nss.py
#
# Description : Computes NSS metric #

# Author : Ming Jiang 

import numpy as np
import scipy.ndimage


class SAUC():
    '''
    Class for computing NSS score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs


    def calc_score(self, gtsAnn, resAnn, shufMap, stepSize=.1, Nsplits=10):
        """
        Computer SAUC score. A simple implementation
        :param gtsAnn : list of fixation annotataions
        :param resAnn : list only contains one element: the result annotation - predicted saliency map
        :return score: int : score
        """
        #result should be only one saliency map
        image_id = resAnn[0]['image_id']
        #get ground truth fixations and result saliency map
        fixations = [ ann['fixations'] for ann in gtsAnn ] # fixations list
        merged_fixations = [item for sublist in fixations for item in sublist]
        salmap = self.saliconRes.decodeImage(resAnn[0]['saliency_map'])
        
        #get size of the original image
        height,width = (self.imgs[image_id]['height'],self.imgs[image_id]['width'])
        mapheight,mapwidth = np.shape(salmap)
        salmap = scipy.ndimage.zoom(salmap, (float(height)/mapheight, float(width)/mapwidth), order=3)
        salmap = (salmap - np.min(salmap))/(np.max(salmap) - np.min(salmap))
        
        S = salmap.reshape(-1)
        F = self.saliconRes.buildFixMap(gtsAnn, False)
        Sth = np.asarray([ salmap[y-1][x-1] for y,x in merged_fixations ])
        
        Nfixations = len(merged_fixations)
        Npixels = len(S)
        
        # for each fixation, sample Nsplits values from anywhere on the sal map
        r = np.random.randint(Npixels, size=(Nfixations,Nsplits))

        #randfix = nan(Nfixations,Nsplits)
        others = (shufMap-F)>0
        ind = np.nonzero(others) # find fixation locations on other images
        Nothers = len(ind[0])
        
        # randomize choice of fixation locations
        randfix = [[salmap[ind[0][k]][ind[1][k]] for k in np.random.choice(Nothers, Nfixations, replace=False)]\
                   for i in range(Nsplits)]
        
        # calculate AUC per random split (set of random locations)
        auc = np.full(Nsplits, np.nan)
        
        for s in range(Nsplits):
            curfix = randfix[s]
            allthreshes = np.arange(0,np.max(np.concatenate((Sth, curfix), axis=0)),stepSize)
            allthreshes = allthreshes[::-1]
            tp = np.zeros(len(allthreshes)+2)
            fp = np.zeros(len(allthreshes)+2)
            tp[-1]=1.0
            fp[-1]=1.0
            tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
            fp[1:-1]=[float(np.sum(curfix >= thresh))/Nfixations for thresh in allthreshes]

            auc[s] = np.trapz(tp,fp)

        return np.mean(auc)

    def compute_score(self, gts, res, shufMap=np.zeros((480,640))):
        """
        Computes SAUC score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : salmap predictions with "image name" key and ndarray as values 
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = res.keys()
        score = []
        all_fixations = []
                      
        # we assume all image sizes are 640x480
        for id in imgIds:
            gtsAnn  = gts[id]
            F = self.saliconRes.buildFixMap(gtsAnn, False)
            shufMap += F
        
        shufMap = shufMap>0
        
        for id in imgIds:
            salmap = res[id]
            fixations  = gts[id]
            score.append(self.calc_score(fixations,salmap,shufMap))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "SAUC"

   

if __name__=="__main__": 
    nss = SAUC()
    #more tests here
