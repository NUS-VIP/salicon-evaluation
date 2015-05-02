#!/usr/bin/env python
#
# File Name : auc.py
#
# Description : Computes AUC metric

# Author : Ming Jiang

import numpy as np
import scipy.ndimage


class AUC():
    '''
    Class for computing NSS score for saliency maps

    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs


    def calc_score(self, gtsAnn, resAnn, stepSize=.01, Nrand=100000):
        """
        Computer AUC score.
        :param gtsAnn : ground-truth annotations
        :param resAnn : predicted saliency map
        :return score: int : score
        """

        salMap = resAnn - np.min(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.max(salMap)

        S = salMap.reshape(-1)
        Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ])

        Nfixations = len(gtsAnn)
        Npixels = len(S)

        # sal map values at random locations
        randfix = S[np.random.randint(Npixels, size=Nrand)]

        allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
        allthreshes = allthreshes[::-1]
        tp = np.zeros(len(allthreshes)+2)
        fp = np.zeros(len(allthreshes)+2)
        tp[-1]=1.0
        fp[-1]=1.0
        tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
        fp[1:-1]=[float(np.sum(randfix >= thresh))/Nrand for thresh in allthreshes]

        auc = np.trapz(tp,fp)
        return auc

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
            img = self.imgs[id]
            fixations  = gts[id]
            height,width = (img['height'],img['width'])
            salMap = self.saliconRes.decodeImage(res[id])
            mapheight,mapwidth = np.shape(salMap)
            salMap = scipy.ndimage.zoom(salMap, (float(height)/mapheight, float(width)/mapwidth), order=3)
            score.append(self.calc_score(fixations,salMap))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "AUC"



if __name__=="__main__":
    nss = AUC()
    #more tests here
