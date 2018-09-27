# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa
from feature_summary import *

from sklearn.linear_model import SGDClassifier

from sklearn.mixture import GaussianMixture

def predict(gmms, test_file):
    scores = []
    for gmm_name, gmm in gmms.items():
        scores.append((gmm_name, gmm.score([test_file])))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def evaluate(gmms, valid_X, valid_Y):
    correct = 0

    for idx, valid_x in enumerate(valid_X):
        valid_y_hat, score = predict(gmms, valid_x)[0]
        #print 'Ground Truth: %s, Predicted: %s, Score: %f' % (valid_Y[idx], valid_y_hat, score)
        if valid_Y[idx] == valid_y_hat:
            correct += 1

    print 'Overall Accuracy: %f%%' % (float(correct) / valid_Y.shape[0]*100.0)


def train_model(train_X, train_Y):

    # Choose a classifier (here, linear SVM)
    #clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)

    clf = GaussianMixture(n_components=64, n_init=20)
    clf.fit(train_X)


    return clf

if __name__ == '__main__':
    gmms = {}
    # load data
    train_X = mean_mfcc('train')
    valid_X = mean_mfcc('valid')
    test_X = mean_mfcc('test')

    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 100)
    valid_Y = np.repeat(cls, 20)
    test_Y = np.repeat(cls, 20)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    # training model
    model = []
    valid_acc = []
    for a in cls:
        gmms[a] = train_model(train_X[train_Y==a], train_Y[train_Y==a])

    evaluate(gmms, valid_X, valid_Y)
    # choose the model that achieve the best validation accuracy
    #final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X/(train_X_std + 1e-5)
    #test_Y_hat = gmms.predict(test_X)
    evaluate(gmms, test_X, test_Y)
    #accuracy = np.sum((test_Y_hat == test_Y))/200.0*100.0
    #print 'test accuracy = ' + str(accuracy) + ' %'

