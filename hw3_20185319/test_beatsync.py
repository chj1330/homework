'''
main_beatsync.py

A file for training frame level chroma and exporting.
Please check the MODE and DEVICE first before you run this code.
Following three lines are different compared to 'main_frame.py'.

 MODE = 'beatsync'
 EXPORT_DIR = './export/baseline_beatsync_result/'
 acc_test, pred_test = data_manager.frame_accuracy(chord_test, pred_test, info_test, BATCH_SIZE, mode=MODE)
'''
import os
import argparse
import numpy as np
import torch
import data_manager
from model_wrapper import Wrapper

def main():
    #Directory Settings
    DATASET_DIR = './dataset/'
    EXPORT_DIR = './export/result/'

    #Parameter Settings
    MODE = 'beatsync'
    DEVICE = 1 # 0 : cpu, 1 : gpu0, 2 : gpu1, ...
    NUM_CLASS = 25 # 0 : Silence, 1 - 12: Major, 13 - 24: Minor, Don't change this parameter
    EPOCH = 100
    BATCH_SIZE = 128
    LEARN_RATE = 0.001
    SEQ_LENGTH = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, default=EXPORT_DIR, help='export directory')
    parser.add_argument('--mode', type=str, default=MODE, help='which mode? frame or beatsync')
    parser.add_argument('--device', type=int, default=DEVICE, help='which device? 0 : cpu, over 1 : gpu')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='how many epoch?')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='how many batch?')
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE, help='learning rate')
    parser.add_argument('--seq_length', type=int, default=SEQ_LENGTH, help='CNN sequence length')
    args = parser.parse_args()
    EXPORT_DIR = args.export_dir
    MODE = args.mode
    DEVICE = args.device
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LEARN_RATE = args.learn_rate
    SEQ_LENGTH = args.seq_length

    #Preprocess
    x, y, info_test = data_manager.preprocess(DATASET_DIR, BATCH_SIZE, SEQ_LENGTH, mode=MODE)
    total_batch = float(x.train.shape[0] + x.test.shape[0] + x.valid.shape[0])
    print('Data Loaded\n'
        + 'Train Ratio : ' + str(round(100*x.train.shape[0]/total_batch, 2))
        + '%, Test Ratio : ' + str(round(100*x.test.shape[0]/total_batch, 2))
        + '%, Valid Ratio : ' + str(round(100*x.valid.shape[0]/total_batch, 2)) + '%')


    #Train
    print('\n--------- Training Start ---------')
    wrapper = Wrapper(x.train.shape[-1], NUM_CLASS, LEARN_RATE)
    #wrapper.model.cuda(device=DEVICE-1)
    # x = minibatch x batchsize x chroma // y = minibatch x batchsize


    #Load model
    model = torch.load('export/model_56.489192.pth')
    wrapper.model = model   

    #Test
    pred_test, _, _ = wrapper.run_model(x.test, y.test, DEVICE, 'eval')

    chroma_test = data_manager.batch_dataset(info_test.chroma, BATCH_SIZE)
    chord_test = data_manager.batch_dataset(info_test.chord, BATCH_SIZE)
    chroma_test = chroma_test.reshape(chroma_test.shape[0] * chroma_test.shape[1], chroma_test.shape[-1])
    chord_test = chord_test.reshape(chord_test.shape[0] * chord_test.shape[1])

    acc_test, pred_test = data_manager.frame_accuracy(chord_test, pred_test, info_test, BATCH_SIZE, mode=MODE)
    print('\nTest Accuracy : ' + str(round(100 * acc_test, 2)) + '%')

    # Export
    wrapper.export(EXPORT_DIR, chroma_test, chord_test, pred_test, acc_test)
    print('Exported files to ' + os.path.abspath(EXPORT_DIR))


if __name__ == '__main__':
    main()
