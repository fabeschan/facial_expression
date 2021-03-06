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
import run_knn, skLearnStuff
import time
import sys

def init_data():

    # Training Data
    imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
    tr_images = imagesDic["tr_images"].astype(float)
    tr_identity = imagesDic["tr_identity"].astype(float)
    tr_labels = imagesDic["tr_labels"]
    tr_images_O = tr_images
    tr_identity_O = tr_identity
    tr_labels_O = tr_labels


    # Test Data
    imagesDic = scipy.io.loadmat(file_name="public_test_images.mat")
    test_images = imagesDic["public_test_images"].astype(float)
    test_images_O = test_images
    
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

    # More processing
    if False:
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
        tr_images = np.array([filters.gaussian_laplace(tr_images[:,:,i], sigma=0.4) for i in xrange(tr_images.shape[2])])
        test_images = np.array([filters.gaussian_laplace(test_images[:,:,i], sigma=0.4) for i in xrange(test_images.shape[2])])

        #tr_images = np.array([filter.edges.prewitt(tr_images[:,:,i]) for i in xrange(tr_images.shape[2])])
        #test_images = np.array([filter.edges.prewitt(test_images[:,:,i]) for i in xrange(test_images.shape[2])])
        #tr_images = np.array([filter.edges.sobel(tr_images[:,:,i]) for i in xrange(tr_images.shape[2])])
        #test_images = np.array([filter.edges.sobel(test_images[:,:,i]) for i in xrange(test_images.shape[2])])
        tr_images = np.rollaxis(tr_images, 0, 3)
        test_images = np.rollaxis(test_images, 0, 3)
        if SHOW_FILTER:
            plt.figure(2)
            plt.clf()
            plt.imshow(tr_images[:,:,0], cmap=plt.cm.gray)
            plt.show()
            sys.exit()

    if False:
        ID = 1011
        tr_ID = tr_images[:,:,tr_identity.ravel()==ID]
        for i in range(tr_ID.shape[2]):
            plt.figure(i + 3)
            plt.clf()
            plt.imshow(tr_ID[:,:,i], cmap=plt.cm.gray)
            plt.show()
        sys.exit()

    # Preprocess the training set
    tr_images = np.array([tr_images[:,:,i].reshape(-1) for i in xrange(tr_images.shape[2])])
    tr_images = preprocessing.scale(tr_images,1)
    
    tr_images_O = np.array([tr_images_O[:,:,i].reshape(-1) for i in xrange(tr_images_O.shape[2])])
    tr_images_O = preprocessing.scale(tr_images_O,1)

    # Preprocess the test set
    test_images = np.array([test_images[:,:,i].reshape(-1) for i in xrange(test_images.shape[2])])
    test_images = preprocessing.scale(test_images,1)
    
    test_images_O = np.array([test_images_O[:,:,i].reshape(-1) for i in xrange(test_images_O.shape[2])])
    test_images_O = preprocessing.scale(test_images_O,1)

    # PCA reduction/projection
    if False:
        dim = 250
        pca = PCA(n_components=dim)
        tr_images = pca.fit_transform(tr_images)

        pca_ = PCA(n_components=dim)
        test_images = pca.fit_transform(test_images)
        print "PCA Total explained variance:", np.sum(pca.explained_variance_ratio_)

    return tr_images, tr_labels, tr_identity, test_images, tr_images_O, tr_labels_O, tr_identity_O, test_images_O

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

def validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=5, verbose=True, aux_classify1=False, aux_classify2=False):
    ''' Do n-fold validation with multiple classifiers; based on popularity vote '''

    j = 1
    valid_score = 0.0
    Histogram1 = {}
    Histogram2 = {}
    
    trained_6 = aux_classify1
    trained_7 = aux_classify2
    
    #score, pred = validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=6)
    
    
    for nfold_tr_images, nfold_tr_labels, nfold_val_images, nfold_val_labels, nfold_val_identity in cross_validations(tr_images, tr_labels, tr_identity, nFold=nFold):
        prediction_ensemble = []
        for c in classifiers:

            trained = c.fit(nfold_tr_images, nfold_tr_labels)
            pred = trained.predict(nfold_val_images).ravel()
            prediction_ensemble.append(pred)

        prediction_ensemble = np.array(prediction_ensemble)
        pred_voted = stats.mode(prediction_ensemble, axis=0)[0].ravel().astype(int)

        #create histogram 
        labelPairs = zip(pred_voted.tolist(), nfold_val_labels.tolist())

        for val in labelPairs:
            Histogram1[val] = Histogram1.get(val,0) + 1
            
        #use auxilary classifier to classify tricky cases
        #if aux_classify1 and aux_classify2:
        if False:
            pred_6_index = np.nonzero(pred_voted == 6)
            re_eval = trained_6.predict(nfold_val_images[pred_6_index])
            for i in range(len(pred_6_index)):
                pred_voted[pred_6_index[i]] = re_eval[i]
            
            pred_7_index = np.nonzero(pred_voted == 7)
            re_eval = trained_7.predict(nfold_val_images[pred_7_index])
            for i in range(len(pred_7_index)):
                pred_voted[pred_7_index[i]] = re_eval[i]         
            
            #create histogram for corrected labels
            labelPairs = zip(pred_voted.tolist(), nfold_val_labels.tolist())
    
            for val in labelPairs:
                Histogram2[val] = Histogram2.get(val,0) + 1
            
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

    #unidentified = d[-1]
    #del d[-1]

    #for i in range(len(unidentified)):
    #    index = d.keys()[i % len(d.keys())]
    #    d[index].append(unidentified[i])

    #create folds
    folds = cross_validation.KFold(len(d.keys()), nFold, shuffle=True)
    d_values = d.values()

    #convert fold randomization into usable indicies
    for train_index, test_index in folds:

        tr_images = []
        tr_labels = []
        val_images = []
        val_labels = []
        val_identity = []

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
    tr_images, tr_labels, tr_identity, test_images, tr_images_O, tr_labels_O, tr_identity_O, test_images_O = init_data()

    #labels = run_knn.run_knn(5, train_img, train_labels, train_img)


    knn_bagging = BaggingClassifier(
        neighbors.KNeighborsClassifier(n_neighbors=5, p=2),
        n_estimators=45,
        max_samples=0.3,
        max_features=0.4,
        bootstrap_features=True,
        n_jobs=8,
        )

    svm_bagging = BaggingClassifier(
        svm.LinearSVC(),
        n_estimators=40,
        n_jobs=8,
        max_samples=0.4,
        max_features=0.4,
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

    gnb_ada = BaggingClassifier(
        linear_model.LogisticRegression(C=.01),
        n_estimators=30,
        n_jobs=8,
        max_samples=0.5,
        max_features=0.5,
        )

    classifiers = [
        #neighbors.KNeighborsClassifier(n_neighbors=7, p=2),
        svm.SVC(),
        #svm.LinearSVC(),
        #svm_bagging,
        #OneVsRestClassifier(svm.LinearSVC(random_state=0)),
        #OutputCodeClassifier(svm.LinearSVC(random_state=0), code_size=2, random_state=0),
        #knn_bagging,
        #linear_model.LogisticRegression(C=.01),
        #linear_model.RidgeClassifierCV(),
        #gnb_ada,
        #naive_bayes.GaussianNB(),
        #tree.DecisionTreeClassifier(criterion="entropy"),
        #trees_bagging,
        #RandomForestClassifier(n_estimators=150),
        #AdaBoostClassifier(n_estimators=100),
        #bdt_real,
        #adaboost,
    ]

    if False:
        nCluster = 15
        #kmean = cluster.KMeans(n_clusters=nCluster, n_jobs=8)
        kmean = cluster.MiniBatchKMeans(n_clusters=nCluster, batch_size=30, n_init=12)
        train_clusters = kmean.fit_predict(tr_images)
        test_clusters = kmean.predict(test_images)

        clusters = []
        scores = []
        final_score = 0
        pred_counter = 0
        for i in range(nCluster):

            #Clustering
            cluster_index = train_clusters==i
            cluster_images = tr_images[cluster_index]
            cluster_labels = tr_labels[cluster_index]
            cluster_identity = tr_identity[cluster_index]
            #PCA
            if True:
                dim = 50
                pca = PCA(n_components=dim)
                tr_images = pca.fit_transform(tr_images)

                pca_ = PCA(n_components=dim)
                test_images = pca.fit_transform(test_images)
                print "PCA Total explained variance:", np.sum(pca.explained_variance_ratio_)
            start = time.time()
            score, pred = validate_multiple(classifiers, cluster_images, cluster_labels, cluster_identity, nFold=6, verbose=False)
            scores.append(score)
            pred_counter = pred_counter + pred.size
            end = time.time()
            elapsed = end - start
            print "Time taken: ", elapsed, "seconds."
            print "cluster size:", cluster_labels.size

        #calculate standard dev
        scores= np.array(scores)

        print "Overall rate:", final_score/float(pred_counter)
        print "range:", np.max(scores) - np.min(scores)
    else:
        #pred_voted = generate_test_labels(classifiers, tr_images, tr_labels, test_images)
        #write_to_file(pred_voted)
        if False:
            subset = np.logical_or((tr_labels == 6),(tr_labels == 3))
            subset = np.nonzero(subset)[0]
            tmp_images = tr_images[subset]
            tmp_labels = tr_labels[subset]
            tmp_identity = tr_identity[subset]
            
            #score, pred = validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=6)
            
            validate_multiple(classifiers, tmp_images, tmp_labels, tmp_identity, nFold=6)

                        
            print "done"
        else:
                       
            subset = np.logical_or((tr_labels == 6),(tr_labels == 3))
            subset = np.nonzero(subset)[0]
            tmp_images = tr_images[subset]
            tmp_labels = tr_labels[subset]
            trained_6 = svm.SVC().fit(tmp_images, tmp_labels.ravel())
            
            subset = np.logical_or((tr_labels == 7),(tr_labels == 5))
            subset = np.nonzero(subset)[0]
            tmp_images = tr_images[subset]
            tmp_labels = tr_labels[subset]
            trained_7 = svm.SVC().fit(tmp_images, tmp_labels.ravel())
            
           
            start = time.time()
            validate_multiple(classifiers, tr_images_O, tr_labels_O, tr_identity_O, nFold=6, aux_classify1 = trained_6, aux_classify2 = trained_7)
            #validate_multiple(classifiers, tr_images, tr_labels, tr_identity, nFold=6)
            end = time.time()
            elapsed = end - start
            print "Time taken: ", elapsed, "seconds."
            """                
            K = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2]
            scores = []
            for  k in K:
                score, pred = validate_multiple([svm.SVC(C=k)], tr_images, tr_labels, tr_identity, nFold=6)                    
                scores.append(score)
            svmResult = zip(K, score)
            print svmResult
            
            K = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
            scores = []
            for  k in K:
                score, pred = validate_multiple([svm.SVC(kernel=k)], tr_images, tr_labels, tr_identity, nFold=6)                    
                scores.append(score)
            svm2Result = zip(K, score)
            print svm2Result
            
            K = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2]
            K = np.array(K)
            K = 1/(2*K)
            
            scores = []
            score, pred = validate_multiple([linear_model.RidgeClassifierCV(alphas = K)], tr_images, tr_labels, tr_identity, nFold=6)                    
            scores.append(score)
            ridgeResult = zip(K, score)
            print ridgeResult
            
            """           

