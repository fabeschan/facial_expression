import scipy.io
import pickle
import numpy as np
from scipy.ndimage import filters
from sklearn import preprocessing

def init_data():

    # Test Data
    imagesDic = scipy.io.loadmat(file_name="public_test_images.mat")
    test_images = imagesDic["public_test_images"].astype(float)

    # Preprocess the test set
    test_images = np.array([filters.gaussian_laplace(test_images[:,:,i], sigma=[0.6, 0.6], mode='reflect') for i in xrange(test_images.shape[2])])
    test_images = np.rollaxis(test_images, 0, 3)

    test_images = np.array([test_images[:,:,i].reshape(-1) for i in xrange(test_images.shape[2])])
    test_images = preprocessing.scale(test_images,1)

    return test_images

if __name__ == '__main__':
    test_images = init_data()
    trained_model = file('trainedModel.pkl', 'r')
    trained_classifier = pickle.load(trained_model)
    predictions = trained_classifier.predict(test_images)
    
    data = {}
    data['predictions'] = predictions
    scipy.io.savemat('test.mat',data)
    trained_model.close()


    #results = scipy.io.loadmat(file_name="test.mat")
    #print results