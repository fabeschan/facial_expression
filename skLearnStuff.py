from sklearn import datasets, neighbors, linear_model, svm
import numpy as np

class data:
    data = None
    target = None

'''
def train_classifier(classifier=None):
    sdata = data()
    sdata.data, sdata.target = generate()
    digits = sdata

    X_digits = digits.data
    y_digits = digits.target

    n_samples = len(X_digits)

    # data
    X_train = X_digits[:]
    y_train = y_digits[:]

    if not classifier:
        classifier = svm.NuSVC(nu=0.01, probability=True)
        #classifier = linear_model.RidgeClassifierCV()

    classifier_fit = classifier.fit(X_train, y_train)
    return classifier_fit
'''

def test(d, t, classifier=None, verbose=True):
    sdata = data()
    sdata.data, sdata.target = d, t
    digits = sdata

    X_digits = digits.data
    y_digits = digits.target

    n_samples = len(X_digits)

    # data
    X_train = X_digits[:.85 * n_samples]
    y_train = y_digits[:.85 * n_samples]

    # truths/target
    X_test = X_digits[.85 * n_samples:]
    y_test = y_digits[.85 * n_samples:]

    if not classifier:
        classifier = linear_model.RidgeClassifierCV()

    classifier_fit = classifier.fit(X_train, y_train)

    pred = classifier_fit.predict(X_test)
    score = classifier_fit.score(X_test, y_test)
    #false_pos = np.sum([ 1 if y_test[i] == 0 and pred[i] == 1 else 0 for i in xrange(len(y_test)) ])
    #false_neg = np.sum([ 1 if y_test[i] == 1 and pred[i] == 0 else 0 for i in xrange(len(y_test)) ])
    #true_pos, true_neg = np.sum(y_test), np.sum(1 - y_test)

    if verbose:
        print 'TRUTH:', y_test
        print 'PREDN:', pred
        print ('Classifier score: %f' % score)
        #print 'Positives: (True / False) {} / {}'.format(true_pos, false_pos)
        #print 'Negatives (True / False) {} / {}'.format(true_neg, false_neg)

    #return score, pred, y_test, false_pos, false_neg, true_pos, true_neg
    return score, pred, y_test
