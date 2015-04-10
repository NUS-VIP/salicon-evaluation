__author__ = 'shane-huang'
__version__ = '1.0'
# Interface for accessing the SALICON dataset - saliency annotations for Microsoft COCO dataset.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load result file and create result api object.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.

import json
import sys
sys.path.append("../")
from pycocotools.coco import COCO



class SALICON (COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of SALICON helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :return:
        """
        COCO.__init__(self,annotation_file=annotation_file)
       

    def createIndex(self):
        """
        Didn't change the original method, just call super
        """
        return COCO.createIndex(self)

    def info(self): 
        """
        Didn't change the original method, just call super
        """
        return COCO.info(self)

    def getAnnIds(self, imgIds=[]):
        return self.getAnnIds(imgIds)

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats (must be empty)
               areaRng (float array)   : get anns for given area range (e.g. [0 inf]) (must be empty)
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catIds) != 0 or len(areaRng) != 0:
            print "Error: does not support category or area range filtering in saliency annoations!"
            return []

        if len(imgIds) == 0:
            anns = self.dataset['annotations']
        else:
            anns = sum([self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns],[])
        
        if self.dataset['type'] == 'fixations':
            ids = [ann['id'] for ann in anns]
        else:
            print "Unkonwn dataset type"
            ids = []
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Didn't change the original method, just call supe
        """
        #not support category filtering
        if len(catIds) !=0 :
            return []
        return COCO.getImgIds(self,imgIds,catIds)

    def loadAnns(self, ids=[]):
        """
        Didn't change the default behavior, just call super
        """
        return COCO.loadAnns(self,ids)

    def loadImgs(self, ids=[]):
        """
        Didn't change the original function, just call super
        """
        return COCO.loadImgs(self,ids)

    def showAnns(self, anns):
        """
        TODO: Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        
        if self.dataset['type'] == 'fixations':
            pass
        

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = SALICON()
        res.dataset['images'] = [img for img in self.dataset['images']]
        res.dataset['info'] = copy.deepcopy(self.dataset['info'])
        res.dataset['type'] = copy.deepcopy(self.dataset['type'])
        res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

        print 'Loading and preparing results...     '
        time_t = datetime.datetime.utcnow()
        anns    = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['area']=sum(ann['segmentation']['counts'][2:-1:2])
                ann['bbox'] = []
                ann['id'] = id
                ann['iscrowd'] = 0
        print 'DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds())

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


if __name__ == "__main__":
    s = SALICON('../annotations/fixations_val2014_examples.json')
    s.info()
    print s.getImgIds()
    print s.getAnnIds(imgIds=102625)
