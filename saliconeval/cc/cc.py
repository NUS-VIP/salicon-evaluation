#!/usr/bin/env python
#
# File Name : nss.py
#
# Description : Computes NSS metric #

# Author : Ming Jiang

import numpy as np
import scipy.ndimage


class CC():
    '''
    Class for computing NSS score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs


    def calc_score(self, gtsAnn, resAnn):
        """
        Computer CC score. A simple implementation
        :param gtsAnn : list of fixation annotataions
        :param resAnn : list only contains one element: the result annotation - predicted saliency map
        :return score: int : score
        """

        image_id = resAnn[0]['image_id']
        fixationmap = self.saliconRes.buildFixMap(gtsAnn)
        salmap = self.saliconRes.decodeImage(resAnn[0]['saliency_map'])

        #get size of the original image
        height,width = (self.imgs[image_id]['height'],self.imgs[image_id]['width'])
        mapheight,mapwidth = np.shape(salmap)
        salmap = scipy.ndimage.zoom(salmap, (float(height)/mapheight, float(width)/mapwidth), order=3)
        salmap = (salmap - np.mean(salmap))/np.std(salmap)

        return np.corrcoef(salmap.reshape(-1), fixationmap.reshape(-1))[0][1]

    def compute_score(self, gts, res):
        """
        Computes CC score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : salmap predictions with "image name" key and ndarray as values
        :returns: average_score: float (mean CC score computed by averaging scores for all the images)
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
        return "CC"



if __name__=="__main__":
    nss = CC()
    #more tests here
