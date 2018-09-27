# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import numpy as np



def data_segment(x, y):
    x = np.transpose(x, [0, 2, 1])
    x = np.reshape(x, [-1, 143, 128])
    x = np.transpose(x, [0, 2, 1])
    y = np.repeat(y, 9)
    return x, y

def load_data(label_path, mel_path, melBins, frames):

    # read train / valid / test lists
    y_train_dict = {}
    y_valid_dict = {}
    y_test_dict = {}
    with open(label_path + 'train_filtered.txt') as f:
        train_list = f.read().splitlines()
        for line in train_list:
            y_train_dict[line] = line.split('/')[0]
    with open(label_path + 'valid_filtered.txt') as f:
        valid_list = f.read().splitlines()
        for line in valid_list:
            y_valid_dict[line] = line.split('/')[0]
    with open(label_path + 'test_filtered.txt') as f:
        test_list = f.read().splitlines()
        for line in test_list:
            y_test_dict[line] = line.split('/')[0]


    # labels
    genres = list(set(y_train_dict.values()+y_valid_dict.values()+y_test_dict.values()))
    print(genres)
    for iter in range(len(y_train_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_train_dict[train_list[iter]]:
                y_train_dict[train_list[iter]] = iter2
    for iter in range(len(y_valid_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_valid_dict[valid_list[iter]]:
                y_valid_dict[valid_list[iter]] = iter2
    for iter in range(len(y_test_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_test_dict[test_list[iter]]:
                y_test_dict[test_list[iter]] = iter2

    # load data

    x_train = np.zeros((len(train_list) * 3, melBins, frames))
    y_train = np.zeros((len(train_list) * 3,))
    for iter in range(len(train_list)):
        x_train[iter] = np.load(mel_path[0] + train_list[iter].replace('.wav', '.npy'))
        x_train[iter + len(train_list)] = np.load(mel_path[1] + train_list[iter].replace('.wav', '.npy'))
        x_train[iter + len(train_list) * 2] = np.load(mel_path[2] + train_list[iter].replace('.wav', '.npy'))
        y_train[iter] = y_train_dict[train_list[iter]]
        y_train[iter + len(train_list)] = y_train[iter]
        y_train[iter + len(train_list) * 2] = y_train[iter]

    x_valid = np.zeros((len(valid_list) * 3, melBins, frames))
    y_valid = np.zeros((len(valid_list) * 3,))
    for iter in range(len(valid_list)):
        x_valid[iter] = np.load(mel_path[0] + valid_list[iter].replace('.wav', '.npy'))
        x_valid[iter + len(valid_list)] = np.load(mel_path[1] + valid_list[iter].replace('.wav', '.npy'))
        x_valid[iter + len(valid_list) * 2] = np.load(mel_path[2] + valid_list[iter].replace('.wav', '.npy'))
        y_valid[iter] = y_valid_dict[valid_list[iter]]
        y_valid[iter + len(valid_list)] = y_valid[iter]
        y_valid[iter + len(valid_list) * 2] = y_valid[iter]

    x_test = np.zeros((len(test_list), melBins, frames))
    y_test = np.zeros((len(test_list),))
    for iter in range(len(test_list)):
        x_test[iter] = np.load(mel_path[0] + test_list[iter].replace('.wav', '.npy'))
        y_test[iter] = y_test_dict[test_list[iter]]
    """
    x_train = np.zeros((len(train_list), melBins, frames))
    y_train = np.zeros((len(train_list),))
    for iter in range(len(train_list)):
        x_train[iter] = np.load(mel_path[0] + train_list[iter].replace('.wav', '.npy'))
        y_train[iter] = y_train_dict[train_list[iter]]

    x_valid = np.zeros((len(valid_list), melBins, frames))
    y_valid = np.zeros((len(valid_list),))
    for iter in range(len(valid_list)):
        x_valid[iter] = np.load(mel_path[0] + valid_list[iter].replace('.wav', '.npy'))
        y_valid[iter] = y_valid_dict[valid_list[iter]]


    x_test = np.zeros((len(test_list), melBins, frames))
    y_test = np.zeros((len(test_list),))
    for iter in range(len(test_list)):
        x_test[iter] = np.load(mel_path[0] + test_list[iter].replace('.wav', '.npy'))
        y_test[iter] = y_test_dict[test_list[iter]]
    """
    # normalize the mel spectrograms
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train -= mean
    x_train /= std
    x_valid -= mean
    x_valid /= std
    x_test -= mean
    x_test /= std

    x_train, y_train = data_segment(x_train, y_train)
    x_valid, y_valid = data_segment(x_valid, y_valid)
    x_test, y_test = data_segment(x_test, y_test)

    return x_train,y_train,x_valid,y_valid,x_test,y_test,genres

def load_data2(label_path, mel_path, melBins, frames):

    # read train / valid / test lists
    y_train_dict = {}
    y_valid_dict = {}
    y_test_dict = {}
    with open(label_path + 'train_filtered.txt') as f:
        train_list = f.read().splitlines()
        for line in train_list:
            y_train_dict[line] = line.split('/')[0]
    with open(label_path + 'valid_filtered.txt') as f:
        valid_list = f.read().splitlines()
        for line in valid_list:
            y_valid_dict[line] = line.split('/')[0]
    with open(label_path + 'test_filtered.txt') as f:
        test_list = f.read().splitlines()
        for line in test_list:
            y_test_dict[line] = line.split('/')[0]


    # labels
    genres = list(set(y_train_dict.values()+y_valid_dict.values()+y_test_dict.values()))
    print(genres)
    for iter in range(len(y_train_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_train_dict[train_list[iter]]:
                y_train_dict[train_list[iter]] = iter2
    for iter in range(len(y_valid_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_valid_dict[valid_list[iter]]:
                y_valid_dict[valid_list[iter]] = iter2
    for iter in range(len(y_test_dict)):
        for iter2 in range(len(genres)):
            if genres[iter2] == y_test_dict[test_list[iter]]:
                y_test_dict[test_list[iter]] = iter2

    # load data

    """
    x_train = np.zeros((len(train_list) * 3, melBins, frames))
    y_train = np.zeros((len(train_list) * 3,))
    for iter in range(len(train_list)):
        x_train[iter] = np.load(mel_path[0] + train_list[iter].replace('.wav', '.npy'))
        x_train[iter + len(train_list)] = np.load(mel_path[1] + train_list[iter].replace('.wav', '.npy'))
        x_train[iter + len(train_list) * 2] = np.load(mel_path[2] + train_list[iter].replace('.wav', '.npy'))
        y_train[iter] = y_train_dict[train_list[iter]]
        y_train[iter + len(train_list)] = y_train[iter]
        y_train[iter + len(train_list) * 2] = y_train[iter]

    x_valid = np.zeros((len(valid_list) * 3, melBins, frames))
    y_valid = np.zeros((len(valid_list) * 3,))
    for iter in range(len(valid_list)):
        x_valid[iter] = np.load(mel_path[0] + valid_list[iter].replace('.wav', '.npy'))
        x_valid[iter + len(valid_list)] = np.load(mel_path[1] + valid_list[iter].replace('.wav', '.npy'))
        x_valid[iter + len(valid_list) * 2] = np.load(mel_path[2] + valid_list[iter].replace('.wav', '.npy'))
        y_valid[iter] = y_valid_dict[valid_list[iter]]
        y_valid[iter + len(valid_list)] = y_valid[iter]
        y_valid[iter + len(valid_list) * 2] = y_valid[iter]

    x_test = np.zeros((len(test_list), melBins, frames))
    y_test = np.zeros((len(test_list),))
    for iter in range(len(test_list)):
        x_test[iter] = np.load(mel_path[0] + test_list[iter].replace('.wav', '.npy'))
        y_test[iter] = y_test_dict[test_list[iter]]
    """
    x_train = np.zeros((len(train_list), melBins, frames))
    y_train = np.zeros((len(train_list),))
    for iter in range(len(train_list)):
        x_train[iter] = np.load(mel_path[0] + train_list[iter].replace('.wav', '.npy'))
        y_train[iter] = y_train_dict[train_list[iter]]

    x_valid = np.zeros((len(valid_list), melBins, frames))
    y_valid = np.zeros((len(valid_list),))
    for iter in range(len(valid_list)):
        x_valid[iter] = np.load(mel_path[0] + valid_list[iter].replace('.wav', '.npy'))
        y_valid[iter] = y_valid_dict[valid_list[iter]]


    x_test = np.zeros((len(test_list), melBins, frames))
    y_test = np.zeros((len(test_list),))
    for iter in range(len(test_list)):
        x_test[iter] = np.load(mel_path[0] + test_list[iter].replace('.wav', '.npy'))
        y_test[iter] = y_test_dict[test_list[iter]]

    # normalize the mel spectrograms
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train -= mean
    x_train /= std
    x_valid -= mean
    x_valid /= std
    x_test -= mean
    x_test /= std

    x_train, y_train = data_segment(x_train, y_train)
    x_valid, y_valid = data_segment(x_valid, y_valid)
    x_test, y_test = data_segment(x_test, y_test)

    return x_train,y_train,x_valid,y_valid,x_test,y_test,genres




