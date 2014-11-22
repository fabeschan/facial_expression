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
from sklearn import datasets, neighbors, linear_model, svm, naive_bayes, tree
from sklearn import preprocessing
import run_knn, skLearnStuff

def init_data():

    # Training Data
    imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
    tr_images = imagesDic["tr_images"].astype(float)
    tr_identity = imagesDic["tr_identity"].astype(float)
    tr_labels = imagesDic["tr_labels"]

    # Test Data
    imagesDic = scipy.io.loadmat(file_name="public_test_images.mat")
    test_images = imagesDic["public_test_images"].astype(float)

    # Preprocess the training set
    tr_images = np.array([tr_images[:,:,i].reshape(-1) for i in xrange(tr_images.shape[2])])
    tr_images = preprocessing.scale(tr_images,1)

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

def evaluate_multiple(classifiers, tr_images, tr_labels):

    prediction_ensemble = []
    train_vs_valid_proportion = 0.85
    n_train = train_vs_valid_proportion * len(tr_images)
    for c in classifiers:
        trained = skLearnStuff.train_classifier(tr_images[:n_train], tr_labels[:n_train].ravel(), c, verbose=True)
        score, pred = skLearnStuff.evaluate(tr_images[n_train:], tr_labels[n_train:].ravel(), trained_classifier=trained, verbose=True)
        prediction_ensemble.append(pred)

    prediction_ensemble = np.array(prediction_ensemble)
    pred_voted = np.zeros(prediction_ensemble.shape[1])
    for i in range(prediction_ensemble.shape[1]):
        d = {}
        for k in prediction_ensemble[:,i]:
            t = d.get(k, 0)
            d[k] = t + 1
        pred_voted[i] = max(d.keys(), key=lambda x: d[x])

    print pred_voted
    valid_labels = tr_labels[n_train:]

    false_count = np.flatnonzero(valid_labels.reshape(-1) - pred_voted.reshape(-1)).size
    rate = float(valid_labels.size - false_count)/valid_labels.size
    print "Ensemble classification rate:", rate
    return rate, pred_voted

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

def write_to_file(predictions):
    with open("csv.csv", 'w') as f:
        f.write("id,prediction\n")
        for i in range(len(predictions)):
            s = "{},{}\n".format(i+1, predictions[i])
            f.write(s)

def cross_validations(classifier, images, labels, identity, nFold=10):
    d = {}

    #create dictionary of {identities: indicies of corresponding labels}
    for i in range(len(identity)):
        t = d.get(identity[i][0], [])
        d[identity[i][0]] = t
        t.append(i)
    
    #create folds
    folds = cross_validation.KFold(len(d.keys()), nFold)
    
    scores = []
    
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
        
        trained = classifier.fit(tr_images, tr_labels.ravel())
        
        score = trained.score(val_images, val_labels.ravel())
        scores.append(score)
        
    scores = np.array(scores)
    print scores
    return np.average(scores)
            
        
    
#        images_folds.append(images[d[indicies]])
#        labels_folds.append(labels[d[indicies]])
        
    
if __name__ == '__main__':
    tr_images, tr_labels, tr_identity, test_images = init_data()
    #labels = run_knn.run_knn(5, train_img, train_labels, train_img)

    classifiers = [
        neighbors.KNeighborsClassifier(p=2),
        #linear_model.RidgeClassifierCV(),
        #neighbors.NearestNeighbors(n_neighbors=2, algorithm='ball_tree'),
        #naive_bayes.GaussianNB(),
        #tree.DecisionTreeClassifier(criterion="entropy"),
    ]
    score = cross_validations(classifiers[0], tr_images, tr_labels, tr_identity, nFold = 10)
    print "Validation Score: "+str(score)
    rate, pred_voted = evaluate_multiple(classifiers, tr_images, tr_labels)
    write_to_file(pred_voted)
