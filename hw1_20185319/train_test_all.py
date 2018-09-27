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
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP

def train_model(train_X, train_Y, valid_X, valid_Y, method, hyper_param1):
    if method == 'SGD':
        # Choose a classifier (here, linear SVM)
        clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    elif method =='KNN':
        clf = KNN(n_neighbors=hyper_param1)
    elif method == 'MLP':
        #clf = MLP(hidden_layer_sizes=(50, hyper_param1), batch_size=10, max_iter=2000, beta_1=np.random.uniform(0, 1),
        #          beta_2=np.random.uniform(0, 1))
        clf = MLP(hidden_layer_sizes=(50, hyper_param1), batch_size=10, max_iter=2000, beta_1=0.189229645396,
                  beta_2=0.886844659737)


    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/200.0*100.0
    print method + ' validation accuracy = ' + str(accuracy) + ' %'
    
    return clf, accuracy

if __name__ == '__main__':

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

    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X / (train_X_std + 1e-5)

    model = []
    valid_acc = []

    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, 'SGD', a)
        model.append(clf)
        valid_acc.append(acc)

    alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, 'KNN', a)
        model.append(clf)
        valid_acc.append(acc)


    alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, 'MLP', a)
        model.append(clf)
        valid_acc.append(acc)
    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]


    test_Y_hat = final_model.predict(test_X)

    accuracy = np.sum((test_Y_hat == test_Y))/200.0*100.0
    print 'test accuracy = ' + str(accuracy) + ' %'
    model_filename = 'final_model_' + str(accuracy) + '.pkl'
    joblib.dump(final_model, model_filename)
