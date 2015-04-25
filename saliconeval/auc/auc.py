#!/usr/bin/env python
# 
# File Name : nss.py
#
# Description : Computes NSS metric #

# Author : Ming Jiang 

import numpy as np
import scipy.ndimage


class AUC():
    '''
    Class for computing NSS score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs


    def calc_score(self, gtsAnn, resAnn):
        """
        Computer AUC score. A simple implementation
        :param gtsAnn : list of fixation annotataions
        :param resAnn : list only contains one element: the result annotation - predicted saliency map
        :return score: int : score
        """
        return 0.0

    def compute_score(self, gts, res):
        """
        Computes AUC score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : salmap predictions with "image name" key and ndarray as values 
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = res.keys()
        score = []
        for id in imgIds:
            salmap = res[id]
            fixations  = gts[id]
            score.append(self.calc_score(fixations,salmap))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "AUC"

   

if __name__=="__main__": 
    nss = AUC()
    #more tests here
