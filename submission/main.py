# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:12:02 2014

@author: fabian
@author: jiayi

"""

import scipy.io
from scipy.misc import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from skimage import filter, color, io, exposure
from scipy.ndimage import filters
import pickle

def init_data():

    # Training Data
    imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
    tr_images = imagesDic["tr_images"].astype(float)
    tr_identity = imagesDic["tr_identity"].astype(float)
    tr_labels = imagesDic["tr_labels"]

    # Test Data
    imagesDic = scipy.io.loadmat(file_name="public_test_images.mat")
    test_images = imagesDic["public_test_images"].astype(float)
    imagesDic = scipy.io.loadmat(file_name="hidden_test_images.mat")
    things_to_join = (test_images,imagesDic["hidden_test_images"].astype(float))
    test_images = np.concatenate(things_to_join, axis=2)

    ADD_TRANSFORMED_DATA = True
    if ADD_TRANSFORMED_DATA:
        tr_images_0_1 = transform_(tr_images, 0, 1)
        tr_images_0_m1 = transform_(tr_images, 0, -1)
        tr_images_1_0 = transform_(tr_images, 1, 0)
        tr_images_m1_0 = transform_(tr_images, -1, 0)

        things_to_join = (tr_images, tr_images_0_1, tr_images_0_m1,  tr_images_1_0, tr_images_m1_0)
        tr_images = np.concatenate(things_to_join, axis=2)

        things_to_join = (tr_labels, tr_labels, tr_labels, tr_labels, tr_labels)
        tr_labels = np.concatenate(things_to_join)

        things_to_join = (tr_identity, tr_identity, tr_identity, tr_identity, tr_identity)
        tr_identity = np.concatenate(things_to_join)

    # More processing
    if True:
        tr_images = np.array([filters.gaussian_laplace(tr_images[:,:,i], sigma=[0.5, 0.5], mode='reflect') for i in xrange(tr_images.shape[2])])
        test_images = np.array([filters.gaussian_laplace(test_images[:,:,i], sigma=[0.5, 0.5], mode='reflect') for i in xrange(test_images.shape[2])])

        tr_images = np.rollaxis(tr_images, 0, 3)
        test_images = np.rollaxis(test_images, 0, 3)

    # Preprocess the training set
    tr_images = np.array([tr_images[:,:,i].reshape(-1) for i in xrange(tr_images.shape[2])])
    tr_images = preprocessing.scale(tr_images,1)

    # Preprocess the test set
    test_images = np.array([test_images[:,:,i].reshape(-1) for i in xrange(test_images.shape[2])])
    test_images = preprocessing.scale(test_images,1)

    return tr_images, tr_labels, tr_identity, test_images

def transform_(tr_images, x, y):
    x_width = tr_images.shape[0] + abs(x)
    y_width = tr_images.shape[1] + abs(y)

    # x_a, x_b, y_a, y_b are the limits
    if x >= 0: x_a, x_b = x, x+tr_images.shape[0]
    else: x_a, x_b = 0, tr_images.shape[0]

    if y >= 0: y_a, y_b = y, y+tr_images.shape[1]
    else: y_a, y_b = 0, tr_images.shape[1]

    r = np.array([ imresize(tr_images[:,:,i], (x_width,y_width))[x_a:x_b,y_a:y_b] for i in xrange(tr_images.shape[2]) ])
    return np.rollaxis(r, 0, 3)

def generate_test_labels(classifiers, tr_images, tr_labels, test_images):
    pred_ensemble_test = []

    i = 1
    for c in classifiers:
        fitted = c.fit(tr_images, tr_labels.ravel())
        pred = fitted.predict(test_images)
        pred_ensemble_test.append(pred)
        print "Finished evaluating {}/{} classifiers".format(i, len(classifiers))
        i += 1

    #write test labels
    pred_ensemble_test = np.array(pred_ensemble_test)
    pred_test_voted = np.zeros(pred_ensemble_test.shape[1])
    for i in range(pred_ensemble_test.shape[1]):
        d = {}
        for k in pred_ensemble_test[:,i]:
            t = d.get(k, 0)
            d[k] = t + 1
        pred_test_voted[i] = max(d.keys(), key=lambda x: d[x])

    return pred_test_voted, fitted

def write_to_file(predictions):
    with open("prediction.csv", 'w') as f:
        f.write("id,prediction\n")
        for i in range(1253):
            if i < len(predictions):
                s = "{},{}\n".format(i+1, int(predictions[i]))
                f.write(s)
            else:
                s = "{},{}\n".format(i+1, 0)
                f.write(s)

if __name__ == '__main__':
    tr_images, tr_labels, tr_identity, test_images = init_data()

    classifiers = [ svm.SVC(C=1.6) ]

    pred_voted, fitted = generate_test_labels(classifiers, tr_images, tr_labels, test_images)
    write_to_file(pred_voted)
    f = file('trainedModel.pkl', 'w')
    pickle.dump(fitted, f)
    f.close()
