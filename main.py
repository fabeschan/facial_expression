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
from sklearn import preprocessing, cluster
from sklearn.decomposition import PCA
import run_knn, skLearnStuff
import random

def init_data():

    # Training Data
    imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
    tr_images = imagesDic["tr_images"].astype(float)
    tr_identity = imagesDic["tr_identity"].astype(float)
    tr_labels = imagesDic["tr_labels"]

    # Test Data
    imagesDic = scipy.io.loadmat(file_name="public_test_images.mat")
    test_images = imagesDic["public_test_images"].astype(float)

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
        tr_images_0_2 = transform_(tr_images, 0, 2)
        tr_images_0_m2 = transform_(tr_images, 0, -2)
        tr_images_2_0 = transform_(tr_images, 2, 0)
        tr_images_m2_0 = transform_(tr_images, -2, 0)

        things_to_join = (tr_images, tr_images_0_1, tr_images_0_m1,  tr_images_1_0, tr_images_m1_0, tr_images_0_2, tr_images_0_m2,  tr_images_2_0, tr_images_m2_0)
        tr_images = np.concatenate(things_to_join, axis=2)

        things_to_join = (tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels, tr_labels)
        tr_labels = np.concatenate(things_to_join)

        things_to_join = (tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity, tr_identity)
        tr_identity = np.concatenate(things_to_join)

    # Preprocess the training set
    tr_images = np.array([tr_images[:,:,i].reshape(-1) for i in xrange(tr_images.shape[2])])
    tr_images = preprocessing.scale(tr_images,1)

    # Preprocess the test set
    test_images = np.array([test_images[:,:,i].reshape(-1) for i in xrange(test_images.shape[2])])
    test_images = preprocessing.scale(test_images,1)

    # PCA reduction/projection
    if True:
        dim = 50
        pca = PCA(n_components=dim)
        tr_images = pca.fit_transform(tr_images)

        pca_ = PCA(n_components=dim)
        test_images = pca.fit_transform(test_images)
        print "PCA Total explained variance:", np.sum(pca.explained_variance_ratio_)

    return tr_images, tr_labels, tr_identity, test_images

"""
nTrainExample = 2162

train_img = tr_images[:nTrainExample,:]
valid_img = tr_images[nTrainExample:,:]

train_labels = tr_labels[:nTrainExample]
valid_labels = tr_labels[nTrainExample:]
"""

"""
plt.figure(1)
plt.clf()
plt.imshow(tr_images[:,:,2], cmap=plt.cm.gray)
plt.show()
"""

def transform_(tr_images, x, y):
    x_width = tr_images.shape[0] + abs(x)
    y_width = tr_images.shape[1] + abs(y)

    # x_a, x_b, y_a, y_b are the limits
    if x >= 0:
        x_a, x_b = x, x+tr_images.shape[0]
    else:
        x_a, x_b = 0, tr_images.shape[0]

    if y >= 0:
        y_a, y_b = y, y+tr_images.shape[1]
    else:
        y_a, y_b = 0, tr_images.shape[1]

    r = np.array([ imresize(tr_images[:,:,i], (x_width,y_width))[x_a:x_b,y_a:y_b] for i in xrange(tr_images.shape[2]) ])
    return np.rollaxis(r, 0, 3)

def validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=5):
    ''' Do n-fold validation with multiple classifiers; based on popularity vote '''

    j = 1
    valid_score = 0.0
    for nfold_tr_images, nfold_tr_labels, nfold_val_images, nfold_val_labels in cross_validations(tr_images, tr_labels, tr_identity, nFold=nFold):
        prediction_ensemble = []
        for c in classifiers:

            trained = c.fit(nfold_tr_images, nfold_tr_labels.ravel())
            pred = trained.predict(nfold_val_images).ravel()
            prediction_ensemble.append(pred)

        prediction_ensemble = np.array(prediction_ensemble)
        pred_voted = stats.mode(prediction_ensemble, axis=0)[0]
        valid_score_ = np.sum(pred_voted.ravel() == nfold_val_labels.ravel()) / float(nfold_val_labels.size)
        valid_score += valid_score_ * (1 - float(nfold_tr_labels.size) / tr_labels.size)
        print "validate_multiple: completed {}/{} folds (fold score: {})".format(j, nFold, valid_score_)
        j += 1

    print "Ensemble classification rate:", valid_score
    return valid_score, pred_voted

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

def train_multiple(classifiers, tr_images, tr_labels):
    pass

def fetch_classifier():
    ''' WIP!! '''
    try:
        from sklearn.externals import joblib
        c = joblib.load('classifier.pkl')
    except Exception, e:
        print e
        print "Retraining classifier..."
        from sklearn.externals import joblib
        #c = analyze.train_classifier()
        joblib.dump(c, 'classifier.pkl')
    return c

def cross_validations(images, labels, identity, nFold=5):

    #create dictionary of {identities: indicies of corresponding labels}
    d = {}
    for i in range(len(identity)):
        t = d.get(identity[i][0], [])
        d[identity[i][0]] = t
        t.append(i)

    #create folds
    folds = cross_validation.KFold(len(d.keys()), nFold, shuffle=True)

    #convert fold randomization into usable indicies
    for train_index, test_index in folds:

        tr_images = []
        tr_labels = []
        val_images = []
        val_labels = []

        imageIndex = [d.values()[i] for i in train_index.tolist()]
        for index in imageIndex:
            tr_images = tr_images + [images[i] for i in index]
            tr_labels = tr_labels + [labels[i] for i in index]

        imageIndex = [d.values()[i] for i in test_index.tolist()]
        for index in imageIndex:
            val_images = val_images + [images[i] for i in index]
            val_labels = val_labels + [labels[i] for i in index]

        tr_images = np.array(tr_images)
        tr_labels = np.array(tr_labels)
        val_images = np.array(val_images)
        val_labels = np.array(val_labels)

        yield tr_images, tr_labels, val_images, val_labels

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

    #labels = run_knn.run_knn(5, train_img, train_labels, train_img)

    knn_bagging = BaggingClassifier(
        neighbors.KNeighborsClassifier(n_neighbors=5, p=2),
        n_estimators=45,
        max_samples=0.3,
        max_features=0.4,
        bootstrap_features=True,
        n_jobs=8,
        )

    trees_bagging = BaggingClassifier(
        tree.DecisionTreeClassifier(criterion="entropy", max_depth=2),
        n_estimators=45,
        max_samples=0.3,
        max_features=0.4,
        bootstrap_features=False,
        n_jobs=8,
        )

    adaboost = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME",
                         n_estimators=200)

    bdt_real = AdaBoostClassifier(
        tree.DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        learning_rate=1,
        algorithm="SAMME.R",
        )

    ridgeCV_ada = AdaBoostClassifier(
        linear_model.RidgeClassifierCV(),
        n_estimators=200,
        learning_rate=1,
        algorithm="SAMME",
        )

    classifiers = [
        #neighbors.KNeighborsClassifier(n_neighbors=7, p=2),
        #svm.SVC(),
        #knn_bagging,
        #linear_model.LogisticRegression(C=.01),
        #linear_model.RidgeClassifierCV(),
        #ridgeCV_ada,
        #naive_bayes.GaussianNB(),
        #tree.DecisionTreeClassifier(criterion="entropy"),
        #trees_bagging,
        #RandomForestClassifier(n_estimators=150),
        #AdaBoostClassifier(n_estimators=100),
        bdt_real,
        #adaboost,
    ]

    if False:
        nCluster = 15
        #kmean = cluster.KMeans(n_clusters=nCluster, n_jobs=8)
        kmean = cluster.MiniBatchKMeans(n_clusters=nCluster, batch_size=30, n_init=12)
        train_clusters = kmean.fit_predict(tr_images)
        test_clusters = kmean.predict(test_images)

        #clusters = []
        #for i in range(nCluster):
        #    index = np.nonzero(train_clusters==i)[0]
        #    clusters.append(index)
        print (train_clusters==0).shape
        tr_images = tr_images[train_clusters==0]
        tr_labels = tr_labels[train_clusters==0]
        print (tr_labels==0).shape
        tr_identity = tr_identity[train_clusters==0]

    #pred_voted = generate_test_labels(classifiers, tr_images, tr_labels, test_images)
    #write_to_file(pred_voted)

    validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=6)

"""
if __name__ == '__main__':
    tr_images, tr_labels, tr_identity, test_images = init_data()
    #labels = run_knn.run_knn(5, train_img, train_labels, train_img)

    classifiers = [
        neighbors.KNeighborsClassifier(),
        #linear_model.RidgeClassifierCV(),
        #neighbors.NearestNeighbors(n_neighbors=2, algorithm='ball_tree'),
        #naive_bayes.GaussianNB(),
        #tree.DecisionTreeClassifier(criterion="entropy"),
    ]

    #pred_voted = generate_test_labels(classifiers, tr_images, tr_labels, test_images)
    #write_to_file(pred_voted)
    K=[3,4,5,6,7,8,9,10,15,20,35,50]
    for k in K:
        classifier = neighbors.KNeighborsClassifier(p=2, n_neighbors=k)
        valid_score = cross_validations(classifier, tr_images, tr_labels, tr_identity, nFold=5)
        print "valid_score: "+str(valid_score) +"k: " + str(k)


    #rate, pred_voted = evaluate_multiple(classifiers, tr_images, tr_labels)
"""
