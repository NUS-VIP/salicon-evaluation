#!/usr/bin/env python
#
# File Name : sauc.py
#
# Description : Computes shuffled AUC metric #

# Author : Ming Jiang

import numpy as np
import scipy.ndimage
class SAUC():
    '''
    Class for computing SAUC score for saliency maps
    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs


    def calc_score(self, gtsAnn, resAnn, shufMap, stepSize=.01):
        """
        Computer SAUC score. A simple implementation
        :param gtsAnn : list of fixation annotataions
        :param resAnn : list only contains one element: the result annotation - predicted saliency map
        :return score: int : score
        """

        salMap = resAnn - np.min(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.max(salMap)
        Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ])
        Nfixations = len(gtsAnn)

        others = np.copy(shufMap)
        for y,x in gtsAnn:
            others[y-1][x-1] = 0

        ind = np.nonzero(others) # find fixation locations on other images
        nFix = shufMap[ind]
        randfix = salMap[ind]
        Nothers = sum(nFix)

        allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
        allthreshes = allthreshes[::-1]
        tp = np.zeros(len(allthreshes)+2)
        fp = np.zeros(len(allthreshes)+2)
        tp[-1]=1.0
        fp[-1]=1.0
        tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
        fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

        auc = np.trapz(tp,fp)
        return auc

    def compute_score(self, gts, res, shufMap=np.zeros((480,640))):
        """
        Computes SAUC score for a given set of predictions and fixations
        :param gtsAnn : ground-truth annotations
        :param resAnn : predicted saliency map
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = res.keys()
        score = []

        # we assume all image sizes are 640x480
        for id in imgIds:
            fixations  = gts[id]
            gtsAnn = {}
            gtsAnn['image_id'] = id
            gtsAnn['fixations'] = fixations
            shufMap += self.saliconRes.buildFixMap([gtsAnn], False)

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
            score.append(self.calc_score(fixations,salMap,shufMap))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "SAUC"



if __name__=="__main__":
    nss = SAUC()
    #more tests here
