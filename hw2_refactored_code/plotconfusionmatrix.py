from __future__ import print_function
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



import sys
import os
import numpy as np
import time
import argparse

from model import *
from data_loader import *
from preprocessing import *
from train import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# gpu_option
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_use', type=int, help='GPU enable')
parser.add_argument('--which_gpu', type=int, help='GPU enable')
args = parser.parse_args()
print(args)

# options
melBins = 128
hop = 512
frames = int(29.9*22050.0/hop)
batch_size = 5
learning_rate = 0.01
num_epochs = 50

# A location where labels and features are located
label_path = '../hw2/gtzan/'
mel_path = ['../hw2/gtzan_mel/', '../hw2/gtzan_mel_pitch_u/', '../hw2/gtzan_mel_pitch_d/']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')







def main():

    # load normalized mel-spectrograms and labels
    x_train, y_train, _, _, x_test,y_test,genres = load_data(label_path, mel_path, melBins, frames)
    print(x_test.shape,y_test.shape)

    # data loader
    test_data = gtzandata(x_test,y_test)
    train_data = gtzandata(x_train, y_train)
    test_loader = DataLoader(test_data, batch_size=9, shuffle=False, drop_last=False)
    train_loader = DataLoader(train_data, batch_size=9, shuffle=False, drop_last=False)

    # load model
    if args.gpu_use == 1:
        model = model_2DCNN_dx().cuda(args.which_gpu)
    elif args.gpu_use == 0:
        model = model_1DCNN_2()
    #print(model)

    # loss function
    criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    # run
    start_time = time.time()
    #checkpoint = torch.load('2DCNN_seg/checkpoint_acc_70.69.pth')
    checkpoint = torch.load('2DCNN_seg/checkpoint_acc_70.69.pth')
    model.load_state_dict(checkpoint['state_dict'])
    best_accuracy = checkpoint['best_accuracy']
    print(best_accuracy)
    #fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs,args)

    print("--- %s seconds spent ---" % (time.time() - start_time))

    # evaluation
    avg_loss, output_all, label_all = eval(model,test_loader,criterion,args)
    prediction = np.concatenate(output_all)
    prediction = prediction.reshape(len(y_test)/9, 9, 10)
    prediction = np.mean(prediction, axis=1)
    prediction = prediction.argmax(axis=1)

    y_label = np.concatenate(label_all)
    y_label = y_label[::9]

    comparison = prediction - y_label
    acc = float(len(comparison) - np.count_nonzero(comparison)) / len(comparison)
    print('Test Accuracy: {:.4f} \n'. format(acc))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_label, prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=genres,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=genres, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


if __name__ == '__main__':
    main()















