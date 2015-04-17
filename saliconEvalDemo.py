
# coding: utf-8

# In[6]:

#get_ipython().magic(u'reload_ext autoreload')
#get_ipython().magic(u'autoreload 2')
#get_ipython().magic(u'matplotlib inline')
from salicon.salicon import SALICON
from saliconeval.eval import SALICONEval
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# In[2]:

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


# In[3]:

dataDir='.'
dataType='val2014'
algName = 'fake'
annFile='%s/annotations/fixations_%s_examples.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= ['%s/results/salmaps_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]


# In[4]:

# create coco object and cocoRes object
salicon = SALICON(annFile)
saliconRes = salicon.loadRes(resFile)


# In[17]:

# create cocoEval object by taking coco and cocoRes
saliconEval = SALICONEval(salicon, saliconRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
saliconEval.params['image_id'] = saliconRes.getImgIds()

# evaluate results
saliconEval.evaluate()


# print output evaluation scores
print "Final Result for each Metric:"
for metric, score in saliconEval.eval.items():
    print '%s: %.3f'%(metric, score)


# In[ ]:



