import numpy as np
import scipy.io

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

print('argggg')

tr_mat = scipy.io.loadmat('labeled_images_new_bilinear.mat')
tr_test = scipy.io.loadmat('public_test_images.mat')

# tr_labels_raw = tr_mat['tr_labels']
# tr_images_raw = tr_mat['tr_images']
# tr_identity_raw = tr_mat['tr_identity']

tr_labels_raw = tr_mat['tr_labels_new']
tr_images_raw = tr_mat['tr_images_new']
tr_identity_raw = tr_mat['tr_identity_new']

tr_labels = np.array(tr_labels_raw)
# tr_labels = np.squeeze(tr_labels)
tr_labels = tr_labels-1
tr_images = np.array(tr_images_raw)
tr_images = tr_images.T
tr_images = np.array([tr_images[:,:,i].reshape(-1) for i in xrange(tr_images.shape[2])])
tr_identity = np.array(tr_identity_raw)
# tr_identity = np.squeeze(tr_identity)

print tr_labels.shape
print tr_labels[0]
print tr_images.shape
print tr_images[0]
print tr_identity.shape
print tr_identity[0]

N, D = tr_images.shape

alldata = ClassificationDataSet(D,1,7)
for n in xrange(N):
    alldata.addSample(tr_images[n], tr_labels[n])

tstdata, trndata = alldata.splitWithProportion( 0.25 )

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 100, trndata.outdim, outclass=SoftmaxLayer )

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.3, verbose=False, weightdecay=0.01)

for i in range(20):
    trainer.trainEpochs(4)
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
