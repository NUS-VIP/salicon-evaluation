#!/usr/bin/env python
# 
# File Name : nss.py
#
# Description : Computes NSS metric #

# Author : Ming Jiang 

import numpy as np
import scipy.ndimage


class NSS():
    '''
    Class for computing NSS score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs


    def calc_score(self, gtsAnn, resAnn):
        """
        Computer NSS score. A simple implementation
        :param gtsAnn : list of fixation annotataions
        :param resAnn : list only contains one element: the result annotation - predicted saliency map
        :return score: int : NSS score
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
        sal_map = scipy.ndimage.zoom(salmap, (float(height)/mapheight, float(width)/mapwidth), order=3)
        sal_map = (salmap - np.mean(salmap))/np.std(salmap)
        return np.mean([ sal_map[y-1][x-1] for y,x in merged_fixations ])

    def compute_score(self, gts, res):
        """
        Computes NSS score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : salmap predictions with "image name" key and ndarray as values 
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = res.keys()
        score = []
        for id in imgIds :
            salmap = res[id]
            fixations  = gts[id]
            score.append(self.calc_score(fixations,salmap))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "NSS"

   

if __name__=="__main__": 
    nss = NSS()
