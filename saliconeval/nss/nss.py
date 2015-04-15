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
    def __init__(self,cocoRes):
        self.cocoRes = cocoRes
        self.imgs = self.cocoRes.imgs


    def calc_score(self, gtsAnn, resAnn):
        """
        Computer NSS score. A simple implementation
        :param sal_map : ndarray: predicted saliency map 
        :param points : list of points: fixations
        :return score: int : NSS score
        """
        #TODO# why is size needed here? can we omit
        image_id = resAnn[0]['image_id']
        #get ground truth fixations and result saliency map
        fixations = [ ann['fixations'] for ann in gtsAnn ] # fixations list
        salmap = reAnn[0]['saliency_map']
        #get size of the original image
        size = (self.imgs[image_id]['width'],self.imgs[image_gid]['height'])
        map_size = np.shape(salmap)
        sal_map = scipy.ndimage.zoom(salmap, (float(size[0])/map_size[0], float(size[1])/map_size[1]), order=3)
        sal_map = (salmap - np.mean(salmap))/np.std(salmap)
        return np.mean(sal_map[points])

    def compute_score(self, gts, res):
        """
        Computes NSS score for a given set of predictions and fixations
        :param nSalmaps : dict : salmap predictions with "image name" key and ndarray as values 
        :param nPoints : dict : fixation points with "image name" key and list of points as values
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = res.keys()

        score = []
        for id in imgIds:
            salmap = res[id]
            fixations  = gts[id]

            score.append(self.calc_score(salmap, fixations))

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "NSS"

   

if __name__=="__main__":
  
    nss = NSS()
    sal_map = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    fixations = [[0, -1, -2],[0, -1, -2]]
    print nss.calc_score(sal_map, fixations)
