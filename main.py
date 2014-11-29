# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:12:02 2014

@author: fabian
@author: jiayi

"""

import scipy.io
from scipy import stats
from scipy.misc import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import datasets, neighbors, linear_model, svm, naive_bayes, tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OutputCodeClassifier, OneVsRestClassifier
from sklearn import preprocessing, cluster
from sklearn.decomposition import PCA
from skimage import filter, color, io, exposure
from scipy.ndimage import gaussian_filter, laplace
from scipy.ndimage import filters
from sklearn.metrics import confusion_matrix
import skLearnStuff
import time
import sys

def init_data():

    # Training Data
    imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
    tr_images = imagesDic["tr_images"].astype(float)
    tr_identity = imagesDic["tr_identity"].astype(float)
    tr_labels = imagesDic["tr_labels"]

    # Test Data
    imagesDic = scipy.io.loadmat(file_name="public_test_images.mat")
    test_images = imagesDic["public_test_images"].astype(float)

    #tr_images = preprocessing.scale(tr_images,1)
    #test_images = preprocessing.scale(test_images,1)

    SHOW_TRANSFORM_COMPARISON = False
    if SHOW_TRANSFORM_COMPARISON:
        trtr = transform_(tr_images, 0, 4)
        plt.figure(1)
        plt.clf()
        plt.imshow(trtr[:,:,0], cmap=plt.cm.gray)
        plt.show()
        plt.figure(2)
        plt.clf()
        plt.imshow(tr_images[:,:,0], cmap=plt.cm.gray)
        plt.show()

    ADD_TRANSFORMED_DATA = False
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

    ADD_TRANSFORMED_DATA_2 = False
    if ADD_TRANSFORMED_DATA_2:
        tr_images_0_1 = transform_(tr_images, 0, 1)
        tr_images_0_m1 = transform_(tr_images, 0, -1)
        tr_images_1_0 = transform_(tr_images, 1, 0)
        tr_images_m1_0 = transform_(tr_images, -1, 0)
        tr_images_0_2 = transform_(tr_images, 1, 1)
        tr_images_0_m2 = transform_(tr_images, 1, -1)
        tr_images_2_0 = transform_(tr_images, -1, -1)
        tr_images_m2_0 = transform_(tr_images, -1, 1)

        things_to_join = (tr_images, tr_images_0_1, tr_images_0_m1,  tr_images_1_0, tr_images_m1_0, tr_images_0_2, tr_images_0_m2,  tr_images_2_0, tr_images_m2_0)
        tr_images = np.concatenate(things_to_join, axis=2)

        things_to_join = (tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels)
        tr_labels = np.concatenate(things_to_join)

        things_to_join = (tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity)
        tr_identity = np.concatenate(things_to_join)

    # More processing
    if True:
        SHOW_FILTER = False
        if SHOW_FILTER:
            plt.figure(1)
            plt.clf()
            plt.imshow(tr_images[:,:,0], cmap=plt.cm.gray)
            plt.show()
        #tr_images = np.array([exposure.equalize_hist(tr_images[:,:,i]) for i in xrange(tr_images.shape[2])])
        #test_images = np.array([exposure.equalize_hist(test_images[:,:,i]) for i in xrange(test_images.shape[2])])
        #tr_images = np.array([gaussian_filter(tr_images[:,:,i], sigma=0.5) for i in xrange(tr_images.shape[2])])
        #test_images = np.array([gaussian_filter(test_images[:,:,i], sigma=0.5) for i in xrange(test_images.shape[2])])
        #tr_images = np.array([filters.gaussian_laplace(tr_images[:,:,i], sigma=[0.5, 0.60], mode='reflect') for i in xrange(tr_images.shape[2])])
        #test_images = np.array([filters.gaussian_laplace(test_images[:,:,i], sigma=[0.5, 0.60], mode='reflect') for i in xrange(test_images.shape[2])])

        #tr_images = np.rollaxis(tr_images, 0, 3)
        #test_images = np.rollaxis(test_images, 0, 3)

        if SHOW_FILTER:
            plt.figure(2)
            plt.clf()
            plt.imshow(tr_images[:,:,0], cmap=plt.cm.gray)
            plt.show()
            sys.exit()

    # Preprocess the training set
    tr_images = np.array([tr_images[:,:,i].reshape(-1) for i in xrange(tr_images.shape[2])])
    #tr_images = preprocessing.scale(tr_images,1)

    # Preprocess the test set
    test_images = np.array([test_images[:,:,i].reshape(-1) for i in xrange(test_images.shape[2])])
    tr_test_images = np.vstack([test_images, tr_images])
    print tr_test_images.shape
    #test_images = preprocessing.scale(test_images,1)
    preproc = preprocessing.scale(tr_test_images,1)
    test_images, tr_images = preproc[:test_images.shape[0],:], preproc[test_images.shape[0]:,:]
    print test_images.shape, tr_images.shape

    # PCA reduction/projection
    if False:
        dim = 250
        pca = PCA(n_components=dim)
        tr_images = pca.fit_transform(tr_images)

        pca_ = PCA(n_components=dim)
        test_images = pca.fit_transform(test_images)
        print "PCA Total explained variance:", np.sum(pca.explained_variance_ratio_)

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

def validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=5, verbose=True):
    ''' Do n-fold validation with multiple classifiers; based on popularity vote '''

    j = 1
    valid_score = 0.0
    total_labels, total_pred = [], []
    for nfold_tr_images, nfold_tr_labels, nfold_val_images, nfold_val_labels, nfold_val_identity in cross_validations(tr_images, tr_labels, tr_identity, nFold=nFold):
        prediction_ensemble = []
        for c in classifiers:

            trained = c.fit(nfold_tr_images, nfold_tr_labels)
            pred = trained.predict(nfold_val_images).ravel()
            prediction_ensemble.append(pred)

        prediction_ensemble = np.array(prediction_ensemble)
        pred_voted = stats.mode(prediction_ensemble, axis=0)[0].ravel().astype(int)
        valid_score_ = np.sum(pred_voted == nfold_val_labels) / float(nfold_val_labels.size)
        valid_score += valid_score_ * (1 - float(nfold_tr_labels.size) / tr_labels.size)
        if verbose:
            print "validate_multiple: completed {}/{} folds (fold score: {})".format(j, nFold, valid_score_)
            print "\tValid fold size: {}".format(nfold_val_labels.size)
            """
            for vid in set(nfold_val_identity.tolist()):
                v_rate = np.sum(pred_voted[nfold_val_identity==vid] == nfold_val_labels[nfold_val_identity==vid])
                print "\tRate for identity({}): {}/{}".format(vid, v_rate, np.sum(nfold_val_identity==vid))
            """

        total_pred.append(pred_voted)
        total_labels.append(nfold_val_labels)
        j += 1

    print "Ensemble classification rate:", valid_score
    return valid_score, pred_voted, total_pred, total_labels

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

    return pred_test_voted

def cross_validations(images, labels, identity, nFold=5):

    #create dictionary of {identities: indicies of corresponding labels}
    d = {}
    for i in range(len(identity)):
        t = d.get(identity[i][0], [])
        d[identity[i][0]] = t
        t.append(i)

    #create folds
    folds = cross_validation.KFold(len(d.keys()), nFold, shuffle=True)
    d_values = d.values()
    d_keys = d.keys()

    #convert fold randomization into usable indicies
    for train_index, test_index in folds:

        tr_images, tr_labels, val_images, val_labels, val_identity = [], [], [], [] ,[]

        imageIndex = [d_values[i] for i in train_index.tolist()]
        for index in imageIndex:
            tr_images = tr_images + [images[i] for i in index]
            tr_labels = tr_labels + [labels[i] for i in index]

        imageIndex = [d_values[i] for i in test_index.tolist()]
        for index in imageIndex:
            val_images = val_images + [images[i] for i in index]
            val_labels = val_labels + [labels[i] for i in index]
            val_identity = val_identity + [identity[i] for i in index]

        tr_images = np.array(tr_images)
        tr_labels = np.array(tr_labels).ravel()
        val_images = np.array(val_images)
        val_labels = np.array(val_labels).ravel()
        val_identity = np.array(val_identity).ravel()

        yield tr_images, tr_labels, val_images, val_labels, val_identity

def write_to_file(predictions):
    with open("csv.csv", 'w') as f:
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

    classifiers = [
        neighbors.KNeighborsClassifier(n_neighbors=5, p=2),
        #svm.SVC(C=1.6),
        #svm.LinearSVC(),
        #linear_model.LogisticRegression(C=.01),
        #linear_model.RidgeClassifierCV(),
        #tree.DecisionTreeClassifier(criterion="entropy"),
        #RandomForestClassifier(n_estimators=150),
    ]

    start = time.time()
    #pred_voted = generate_test_labels(classifiers, tr_images, tr_labels, test_images)
    #write_to_file(pred_voted)
    _, _, y_pred, y_true = validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=8)
    print 'Confusion Matrix:\n', confusion_matrix(np.hstack(y_true), np.hstack(y_pred))
    end = time.time()
    elapsed = end - start
    print "Time taken: ", elapsed, "seconds."

