# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
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
learning_rate = 0.001
num_epochs = 100

# A location where labels and features are located
label_path = '../hw2/gtzan/'
mel_path = ['../hw2/gtzan_mel/', '../hw2/gtzan_mel_pitch_u/', '../hw2/gtzan_mel_pitch_d/']


def main():

    # load normalized mel-spectrograms and labels
    x_train,y_train,x_valid,y_valid,x_test,y_test = load_data(label_path, mel_path, melBins, frames)
    print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_test.shape)

    # data loader    
    train_data = gtzandata(x_train,y_train)
    valid_data = gtzandata(x_valid,y_valid)
    test_data = gtzandata(x_test,y_test)

    test_loader = DataLoader(test_data, batch_size=9, shuffle=False, drop_last=False)

    # load model
    if args.gpu_use == 1:
        cls = model_1DCNN_2().cuda(args.which_gpu)
        pred = model_predict().cuda(args.which_gpu)

    elif args.gpu_use == 0:
        cls = model_1DCNN_2()
        pred = model_predict()
    #print(model)

    # loss function
    criterion1 = nn.NLLLoss()
    criterion2 = nn.CrossEntropyLoss()

    # run
    start_time = time.time()
    checkpoint = torch.load('loss_num100/checkpoint_acc_68.62.pth')
    cls.load_state_dict(checkpoint['state_dict'])
    best_accuracy = checkpoint['best_accuracy']
    print(best_accuracy)
    #fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs,args)

    print("--- %s seconds spent ---" % (time.time() - start_time))

    # evaluation
    avg_loss, output_all, label_all = eval(cls,test_loader,criterion1,args)
    prediction = np.concatenate(output_all)
    prediction = prediction.reshape(len(y_test)/9, 9, 10)
    prediction = np.mean(prediction, axis=1)
    prediction = prediction.argmax(axis=1)

    y_label = np.concatenate(label_all)
    y_label = y_label[::9]

    comparison = prediction - y_label
    acc = float(len(comparison) - np.count_nonzero(comparison)) / len(comparison)
    print('Test Accuracy: {:.4f} \n'. format(acc))

    prediction_train,label_train = get_prediction(train_data, cls, criterion1, args)
    prediction_valid,label_valid = get_prediction(valid_data, cls, criterion1, args)
    prediction_test,label_test = get_prediction(test_data, cls, criterion1, args)

    train_data_prediction = gtzandata(prediction_train, label_train)
    valid_data_prediction = gtzandata(prediction_valid, label_valid)
    test_data_prediction = gtzandata(prediction_test, label_test)

    train_loader_prediction = DataLoader(train_data_prediction, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader_prediction = DataLoader(valid_data_prediction, batch_size=batch_size, shuffle=True, drop_last=False)

    test_loader_prediction = DataLoader(test_data_prediction, batch_size=1, shuffle=False, drop_last=False)

    fit_predict(pred, train_loader_prediction, valid_loader_prediction, criterion2, learning_rate, num_epochs, args)

    avg_loss, output_all, label_all = eval(pred, test_loader_prediction, criterion2, args)

    prediction = np.concatenate(output_all)
    prediction = prediction.reshape(len(prediction), 10)
    # prediction1 = np.mean(prediction, axis=1)
    # prediction1 = prediction.argmax(axis=1)
    prediction1 = prediction.argmax(axis=1)
    # print(prediction)

    y_label = np.concatenate(label_all)
    # print(y_label)

    comparison = prediction1 - y_label
    acc = float(len(comparison) - np.count_nonzero(comparison)) / (len(comparison))
    print('Test Accuracy: {:.4f} \n'.format(acc))

    save_checkpoint({
        'state_dict': pred.state_dict(),
        'best_accuracy': acc
    }, acc)
if __name__ == '__main__':
    main()






