
# coding: utf-8

# In[9]:

#get_ipython().magic(u'reload_ext autoreload')
#get_ipython().magic(u'autoreload 2')
#get_ipython().magic(u'matplotlib inline')
from salicon.salicon import SALICON
from saliconeval.eval import SALICONEval
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# In[10]:

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


# In[11]:

dataDir='.'
dataType='train2014examples'
algName = 'fake'
annFile='%s/annotations/fixations_%s.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= ['%s/results/fixations_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]


# In[12]:

# create coco object and cocoRes object
salicon = SALICON(annFile)
saliconRes = salicon.loadRes(resFile)


# In[13]:

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


# In[6]:

# plot score histogram
saucScores = [eva['SAUC'] for eva in saliconEval.evalImgs]
plt.hist(saucScores)
plt.title('Histogram of SAUC Scores', fontsize=20)
plt.xlabel('SAUC score', fontsize=20)
plt.ylabel('result counts', fontsize=20)
plt.show()


# In[14]:

# save evaluation results to ./results folder
json.dump(saliconEval.evalImgs, open(evalImgsFile, 'w'))
json.dump(saliconEval.eval,     open(evalFile, 'w'))


# In[ ]:



