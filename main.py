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

nTrainExample = 2462

train_img = tr_images[:,:,:nTrainExample]
valid_img = tr_images[:,:,nTrainExample:]

train_labels = tr_labels[:nTrainExample]
valid_labels = tr_labels[nTrainExample:]

"""
plt.figure(1)
plt.clf()
plt.imshow(tr_images[:,:,2], cmap=plt.cm.gray)
plt.show()
"""

labels = run_knn.run_knn(3, train_img, train_labels, valid_img)

false_count = np.flatnonzero(valid_labels - labels).size
rate = float(valid_labels.size - false_count)/valid_labels.size
print(rate)
