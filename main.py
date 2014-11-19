# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:12:02 2014

@author: fabian
@author: jiayi
"""

import scipy.io
import matplotlib.pyplot as plt

imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
tr_images = imagesDic["tr_images"]
tr_identity = imagesDic["tr_identity"]
tr_labels = imagesDic["tr_labels"]

plt.figure(1)
plt.clf()

plt.imshow(tr_images[:,:,0].reshape(32, 32), cmap=plt.cm.gray)
#plt.draw()
plt.show()
