# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
data_path = './dataset/'
mfcc_path = './mfcc/'
rmse_path = './rmse/'
MFCC_DIM = 13

def extract_features(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name

        y, sr = librosa.load(file_path, sr=22050)

        S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)

        D_harmonic, D_percussive = librosa.decompose.hpss(S)

        #Hmag = librosa.amplitude_to_db(D_harmonic)
        #Pmag = librosa.amplitude_to_db(D_percussive)
        D_H = np.abs(D_harmonic) ** 2
        D_P = np.abs(D_percussive) ** 2
        # mel spectrogram (512 --> 40)
        mel_basis = librosa.filters.mel(sr, 1024, n_mels=40)
        mel_H = np.dot(mel_basis, D_H)
        mel_P = np.dot(mel_basis, D_P)
        #log compression
        log_mel_H = librosa.power_to_db(mel_H)
        log_mel_P = librosa.power_to_db(mel_P)

        # mfcc (DCT)
        mfcc_H = librosa.feature.mfcc(S=log_mel_H, n_mfcc=13)
        mfcc_H_delta = librosa.feature.delta(mfcc_H)
        mfcc_H_delta2 = librosa.feature.delta(mfcc_H, order=2)
        mfcc_H = np.concatenate((mfcc_H, mfcc_H_delta, mfcc_H_delta2), axis=0)
        mfcc_H = mfcc_H.astype(np.float32)

        mfcc_P = librosa.feature.mfcc(S=log_mel_P, n_mfcc=13)
        mfcc_P_delta = librosa.feature.delta(mfcc_P)
        mfcc_P_delta2 = librosa.feature.delta(mfcc_P, order=2)
        mfcc_P = np.concatenate((mfcc_P, mfcc_P_delta, mfcc_P_delta2), axis=0)
        mfcc_P = mfcc_P.astype(np.float32)
        # to save the memory (64 to 32 bits)
        file_name = file_name.replace('.wav','.npy')
        save_file_H = mfcc_path + 'Harmonic/'+ file_name
        save_file_P = mfcc_path + 'Percussive/' + file_name

        if not os.path.exists(os.path.dirname(save_file_H)):
            os.makedirs(os.path.dirname(save_file_H))
        if not os.path.exists(os.path.dirname(save_file_P)):
            os.makedirs(os.path.dirname(save_file_P))
        np.save(save_file_H, mfcc_H)
        np.save(save_file_P, mfcc_P)

        rmse = librosa.feature.rmse(S=S)[0]
        save_file_r = rmse_path + file_name
        if not os.path.exists(os.path.dirname(save_file_r)):
            os.makedirs(os.path.dirname(save_file_r))
        np.save(save_file_r, rmse)

        i = i + 1
        if not (i % 10):
            print i

    f.close();

if __name__ == '__main__':
    extract_features(dataset='train')
    extract_features(dataset='valid')
    extract_features(dataset='test')



