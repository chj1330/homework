# GCT634 (2018) HW2
#
# Apr-11-2018: initial version
#
# Jongpil Lee
#

from __future__ import print_function
import sys
import os
import numpy as np
import librosa
import soundfile

# mel-spec options
fftsize = 1024
window = 1024
hop = 512
melBin = 128

# A location where gtzan labels are located
label_path = './gtzan/'

# read train / valid / test lists
with open(label_path + 'train_filtered.txt') as f:
    train_list = f.read().splitlines()
with open(label_path + 'valid_filtered.txt') as f:
    valid_list = f.read().splitlines()
with open(label_path + 'test_filtered.txt') as f:
    test_list = f.read().splitlines()

song_list = train_list+valid_list+test_list
print(len(song_list))

# A location where gtzan dataset is located
load_path = './gtzan/'

# A location where mel-spectrogram would be saved
save_path_u = './gtzan_mel_pitch_u8/'
save_path_d = './gtzan_mel_pitch_d8/'
def main():

    # save mel-spectrograms
    for iter in range(0,len(song_list)):
        file_name = load_path + song_list[iter].replace('.wav','.au')
        save_name_u = save_path_u + song_list[iter].replace('.wav','.npy')
        save_name_d = save_path_d + song_list[iter].replace('.wav','.npy')
        if not os.path.exists(os.path.dirname(save_name_u)):
            os.makedirs(os.path.dirname(save_name_u))
        if not os.path.exists(os.path.dirname(save_name_d)):
            os.makedirs(os.path.dirname(save_name_d))

        if os.path.isfile(save_name_u) == 1:
            print(iter, save_name_u + "_file_already_extracted!")
            continue
        if os.path.isfile(save_name_d) == 1:
            print(iter, save_name_d + "_file_already_extracted!")
            continue

        # STFT
        y,sr = librosa.load(file_name,sr=22050)
        y_u = librosa.effects.pitch_shift(y, sr, n_steps=8.0, bins_per_octave=24)
        y_d = librosa.effects.pitch_shift(y, sr, n_steps=-8.0, bins_per_octave=24)
        #soundfile.write(save_name_u4, y_u4, sr)
        #soundfile.write(save_name_d4, y_d4, sr)
        S_u = librosa.core.stft(y_u, n_fft=fftsize, hop_length=hop, win_length=window)
        S_d = librosa.core.stft(y_d, n_fft=fftsize, hop_length=hop, win_length=window)

        X_u = np.abs(S_u)
        X_d = np.abs(S_d)

        # mel basis
        mel_basis = librosa.filters.mel(sr,n_fft=fftsize,n_mels=melBin)

        # mel basis are multiplied to the STFT
        mel_S_u = np.dot(mel_basis, X_u)
        mel_S_d = np.dot(mel_basis, X_d)

        # log amplitude compression
        mel_S_u = np.log10(1+10*mel_S_u)
        mel_S_u = mel_S_u.astype(np.float32)
        mel_S_d = np.log10(1 + 10 * mel_S_d)
        mel_S_d = mel_S_d.astype(np.float32)

        # cut audio to have 30-second size for all files
        Nframes = int(29.9*22050.0/hop)
        if mel_S_u.shape[1] > Nframes:
            mel_S_u = mel_S_u[:,:Nframes]
        if mel_S_d.shape[1] > Nframes:
            mel_S_d = mel_S_d[:,:Nframes]

        # save file
        print(iter,mel_S_u.shape,save_name_u)
        np.save(save_name_u,mel_S_u)
        print(iter,mel_S_d.shape,save_name_d)
        np.save(save_name_d,mel_S_d)



if __name__ == '__main__':
    main()
