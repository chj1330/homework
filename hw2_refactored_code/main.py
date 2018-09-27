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
batch_size = 20
learning_rate = 0.001
num_epochs = 10000

# A location where labels and features are located
label_path = '../hw2/gtzan/'
mel_path = ['../hw2/gtzan_mel/', '../hw2/gtzan_mel_pitch_u/', '../hw2/gtzan_mel_pitch_d/']
#mel_path = ['../hw2/gtzan_mel/']



def main():

    # load normalized mel-spectrograms and labels
    x_train,y_train,x_valid,y_valid,x_test,y_test,genres = load_data2(label_path, mel_path, melBins, frames)
    print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_test.shape)

    # data loader    
    train_data = gtzandata(x_train,y_train)
    valid_data = gtzandata(x_valid,y_valid)
    test_data = gtzandata(x_test,y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last = False)
    test_loader = DataLoader(test_data, batch_size=9, shuffle=False, drop_last=False)
    save_path = '2DCNN_seg/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load model
    if args.gpu_use == 1:
        model = model_2DCNN_dx().cuda(args.which_gpu)
    elif args.gpu_use == 0:
        model = model_1DCNN_2_dx()
    #print(model)

    # loss function 
    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()

    # run
    start_time = time.time()
    fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs,args)
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


    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_accuracy': acc
    }, acc, save_path)

if __name__ == '__main__':
    for i in range(100):
        main()






