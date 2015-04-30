__author__ = 'shane-huang'
__version__ = '1.0'

from nss.nss import NSS
from sauc.sauc import SAUC
from auc.auc import AUC
from cc.cc import CC
import numpy as np
class SALICONEval:
    def __init__(self, salicon, saliconRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.salicon = salicon
        self.saliconRes = saliconRes
        self.params = {'image_id': salicon.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        ## set up the scorers,
        scorers = [
            (SAUC(self.saliconRes),"SAUC"),
            (AUC(self.saliconRes),"AUC"),
            (NSS(self.saliconRes), "NSS"),
            (CC(self.saliconRes),"CC"),
        ]

        ## add any initialization here
        fixations = {}
        salmaps = {}
        for imgId in imgIds:
            salmaps[imgId] = self.saliconRes.imgToAnns[imgId][0]['saliency_map']
            fixs = [ann['fixations'] for ann in self.salicon.imgToAnns[imgId]]
            fixs = [item for sublist in fixs for item in sublist]
            fixs.sort()
            fixs = np.asarray(fixs)
            dup_ind = np.all(fixs[1:]==fixs[:-1], axis=1)
            fixations[imgId] = fixs[dup_ind].tolist()

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(fixations, salmaps)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
