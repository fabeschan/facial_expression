# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:12:02 2014

@author: fabian
@author: jiayi

"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
import run_knn

imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
tr_images = imagesDic["tr_images"]
tr_identity = imagesDic["tr_identity"]
tr_labels = imagesDic["tr_labels"]

nTrainExample = 2162

train_img = tr_images[:,:,:nTrainExample]
valid_img = tr_images[:,:,nTrainExample:]

train_labels = tr_labels[:nTrainExample]
valid_labels = tr_labels[nTrainExample:]

train_img = np.array([ train_img[:,:,i].reshape(-1) for i in xrange(train_img.shape[2]) ])
valid_img = np.array([ valid_img[:,:,i].reshape(-1) for i in xrange(valid_img.shape[2]) ])

"""
plt.figure(1)
plt.clf()
plt.imshow(tr_images[:,:,2], cmap=plt.cm.gray)
plt.show()
"""

#a = np.array([[2,3],[1,2]])
#b = np.array([[4,3],[3,2]])
#c = np.array([[2,4],[4,1]])
#
#train_img = np.array([a,b,c,a,b,c])
#train_labels = np.array([2,3,4,2,3,4])
#valid_img = np.array([c,a,c,b,b,c])
#train_img = np.array([ train_img[:,:,i].reshape(-1) for i in xrange(train_img.shape[2]) ])
#valid_img = np.array([ valid_img[:,:,i].reshape(-1) for i in xrange(valid_img.shape[2]) ])

labels = run_knn.run_knn(1, train_img.T, train_labels.T, valid_img.T)

false_count = np.flatnonzero(valid_labels - labels).size
rate = float(valid_labels.size - false_count)/valid_labels.size
print(rate)
