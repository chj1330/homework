# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = './dataset/'
mfcc_path = './mfcc/'
rmse_path = './rmse/'

MFCC_DIM = 39 * 2
RMSE_DIM = 173
FEATURE_DIM = MFCC_DIM + RMSE_DIM
#FEATURE_DIM = RMSE_DIM
def mean_mfcc(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(FEATURE_DIM, 1000))
    else:
        mfcc_mat = np.zeros(shape=(FEATURE_DIM, 200))

    i = 0
    for file_name in f:

        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_H_file = mfcc_path +'Harmonic/'+ file_name
        mfcc_H = np.load(mfcc_H_file)
        mfcc_P_file = mfcc_path +'Percussive/'+ file_name
        mfcc_P = np.load(mfcc_P_file)

        rmse_file = rmse_path + file_name
        rmse = np.load(rmse_file)


        feature = np.concatenate((mfcc_H, mfcc_P), axis=0)
        mfcc_mat[:MFCC_DIM,i]= np.mean(feature, axis=1)
        mfcc_mat[MFCC_DIM:,i] = rmse

        i = i + 1

    f.close()

    return mfcc_mat


if __name__ == '__main__':
    train_data = mean_mfcc('train')
    valid_data = mean_mfcc('valid')
    test_data = mean_mfcc('test')

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,3)
    plt.imshow(test_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.show()








