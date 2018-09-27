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

data_path = './dataset/'
mfcc_path = './mfcc/'

MFCC_DIM = 13
def detect_leading_silence(sound, silence_threshold=.001, chunk_size=10):
    # this function first normalizes audio data
    #calculates the amplitude of each frame
    #silence_threshold is used to flip the silence part
    #the number of silence frame is returned.
    #trim_ms is the counter
    trim_ms = 0
    max_num = max(sound)
    sound = sound/max_num
    sound = np.array(sound)
    for i in range(len(sound)):
        if sound[trim_ms] < silence_threshold:
            trim_ms += 1
    return trim_ms

def extract_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print i

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        #print file_path
        y, sr = librosa.load(file_path, sr=22050)


        ##### Method 1
        #mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)
        
        ##### Method 2 
        start_trim = detect_leading_silence(y)
        end_trim = detect_leading_silence(np.flipud(y))

        duration = len(y)
        trimmed_sound = y[start_trim:duration-end_trim]
        # STFT
        S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)

        # power spectrum
        D = np.abs(S)**2

        # mel spectrogram (512 --> 40)
        mel_basis = librosa.filters.mel(sr, 1024, n_mels=40)
        mel_S = np.dot(mel_basis, D)

        #log compression
        log_mel_S = librosa.power_to_db(mel_S)

        # mfcc (DCT)
        mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        mfcc = mfcc.astype(np.float32)    # to save the memory (64 to 32 bits)
        
        ##### Method 3
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        """
        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = mfcc_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc)

    f.close();

if __name__ == '__main__':
    extract_mfcc(dataset='train')                 
    extract_mfcc(dataset='valid')                                  
    extract_mfcc(dataset='test')



