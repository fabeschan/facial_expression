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

imagesDic = scipy.io.loadmat(file_name="labeled_images.mat")
tr_images = imagesDic["tr_images"].astype(float)
tr_identity = imagesDic["tr_identity"].astype(float)
tr_labels = imagesDic["tr_labels"]

imagesDic = scipy.io.loadmat(file_name="public_test_images.mat")
test_images = imagesDic["public_test_images"].astype(float)

tr_images = np.array([tr_images[:,:,i].reshape(-1) for i in xrange(tr_images.shape[2])])
tr_images = preprocessing.scale(tr_images,1)


nTrainExample = 2162

"""
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

#a = np.array([[2,3],[1,2]])
#b = np.array([[4,3],[3,2]])
#c = np.array([[2,4],[4,1]])
#
#train_img = np.array([a,b,c,a,b,c]).T
#train_labels = np.array([2,3,4,2,3,4])
#valid_img = np.array([c,a,c,b,b,c]).T
#train_img = np.array([ train_img[:,:,i].reshape(-1) for i in xrange(train_img.shape[2]) ])
#valid_img = np.array([ valid_img[:,:,i].reshape(-1) for i in xrange(valid_img.shape[2]) ])

#labels = run_knn.run_knn(5, train_img, train_labels, train_img)

classifiers = [
    #linear_model.RidgeClassifierCV(),
    #neighbors.NearestNeighbors(n_neighbors=2, algorithm='ball_tree'),
    neighbors.KNeighborsClassifier(p=2),
    #naive_bayes.GaussianNB(),
    #tree.DecisionTreeClassifier(criterion="entropy"),
]

pred_ensemble = []

for c in classifiers:
    score, pred, y_test = skLearnStuff.test(tr_images, tr_labels.ravel(), classifier=c, verbose=True)
    pred_ensemble.append(pred)

pred_ensemble = np.array(pred_ensemble)
pred_voted = np.zeros(pred_ensemble.shape[1])
for i in range(pred_ensemble.shape[1]):
    d = {}
    for k in pred_ensemble[:,i]:
        t = d.get(k, 0)
        d[k] = t + 1
    pred_voted[i] = max(d.keys(), key=lambda x: d[x])

print pred_voted
valid_labels = tr_labels[0.85 * tr_labels.size:]

false_count = np.flatnonzero(valid_labels.reshape(-1) - pred_voted.reshape(-1)).size
rate = float(valid_labels.size - false_count)/valid_labels.size
print(rate)


def write_to_file(predictions):
    with open("csv.csv", 'w') as f:
        f.write("id,prediction\n")
        for i in range(len(predictions)):
            s = "{},{}\n".format(i+1, predictions[i])            
            f.write(s)


a = ['2','3','4','1','2','3']
write_to_file(pred_voted)