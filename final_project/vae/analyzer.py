from os.path import join
import os
import glob
import librosa
import numpy as np
import lws
from scipy.io import wavfile
import random

load_dir = './dataset/wav'
save_dir = './dataset/spectrogram'
fft_size = 1024
hop_size = 256
min_level_db = -100
ref_level_db = 20
power = 1.4
interval = 10000
preemphasis_factor = 0.97
fs = 22050
train_ratio = 0.8
def list_dir(path):
    ''' retrieve the 'short name' of the dirs '''
    return sorted([f for f in os.listdir(path) if os.path.isdir(join(path, f))])

def list_full_filenames(path):
    ''' return a generator of full filenames '''
    return (join(path, f) for f in os.listdir(path) if not os.path.isdir(join(path, f)))

def idx2onehot(idx_array, num_class):
    return np.eye(num_class)[idx_array.astype(int).reshape(-1)].astype(np.float32)

# Extract / Synthesis
def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #signal /= np.max(np.abs(signal))
    wavfile.write(path, 22050, wav.astype(np.int16))

def spectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)

def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)

def _lws_processor():
    return lws.lws(fft_size, hop_size, mode="speech")

def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, preemphasis_factor)

def inv_preemphasis(x):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x, preemphasis_factor)



def wav2spec(f):
    wav, _ = librosa.load(f, fs)
    wav = wav / np.abs(wav).max() * 0.999
    spectrum = spectrogram(wav).astype(np.float32).T  # fft x frame -> frame(batch) x fft
    return spectrum

def get_emotions():
    emotions = []
    with open('./emotions.tsv', 'r') as f:
        for line in f:
            fn = line.split()
            emotions.append(fn[0])
    return emotions


def main(emotions, mode, list_name):
    load_path = join(load_dir, mode)
    save_path = join(save_dir, mode)
    N = len(glob.glob(join(load_path, 'S*', '*.wav')))
    num_train_size = int(N * train_ratio)
    total_train_list = []
    total_valid_list = []

    for emo in emotions:
        counter = 1
        data_list = []
        total_data_list = []
        path = join(load_path, emo)
        save_path = join(save_dir, mode, emo)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for f in list_full_filenames(path): # f:'ema00001.wav', ...
            print('\rFile {}/{}: {:50}'.format(counter, N, f), end='')
            spectrum = wav2spec(f)
            labels = emotions.index(emo) * np.ones([spectrum.shape[0], 1], np.float32)
            b = os.path.splitext(f)[0]
            _, b = os.path.split(b)
            spectrum = np.concatenate([spectrum, labels], 1) # (frame_size, fft_size+1)
            np.save(join(save_path, '{}'.format(b)), spectrum)
            counter += 1
            data_list.append(('{}.npy'.format(b), spectrum.shape[0]))
            total_data_list.append((join(emo, '{}.npy'.format(b)), spectrum.shape[0]))
            #data_list.append(('{}.npy'.format(b), spectrum.shape[0]))

        data_list.sort()
        total_train_list.sort()
        random.Random(4).shuffle(data_list)


        train_list = data_list[:num_train_size]
        valid_list = data_list[num_train_size:]

        train_list.sort()
        valid_list.sort()
        total_train_list.extend(train_list)
        total_valid_list.extend(valid_list)
        print()
        with open(join(save_dir, mode, emo, 'train.txt'), 'w') as f:
            for item in train_list:
                f.write("%s|%d\n" % (item[0], item[1]))
        with open(join(save_dir, mode, emo, 'valid.txt'), 'w') as f:
            for item in valid_list:
                f.write("%s|%d\n" % (item[0], item[1]))
    with open(join(save_dir, mode, 'train.txt'), 'w') as f:
        for item in total_train_list:
            f.write("%s|%d\n" % (item[0], item[1]))
    with open(join(save_dir, mode, 'valid.txt'), 'w') as f:
        for item in total_valid_list:
            f.write("%s|%d\n" % (item[0], item[1]))
        #f.write("%d,%d\n" % (total_frame, spectrum.shape[1]))


def main_backup2(emotions, mode, list_name):
    load_path = join(load_dir, mode)
    save_path = join(save_dir, mode)
    N = len(glob.glob(join(load_path, '*', '*.wav')))

    Source = emotions[0]
    Target = emotions[1:]
    counter = 1
    data_list = []
    total_frame = 0

    for emo in emotions:
        path = join(load_path, emo)
        save_path = join(save_dir, mode, emo)

    for s in list_dir(load_path):
        path = join(load_path, s)
        save_path = join(save_dir, mode, s)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for f in list_full_filenames(path):  # f:'ema00001.wav', ...
            print('\rFile {}/{}: {:50}'.format(counter, N, f), end='')
            spectrum = wav2spec(f)
            labels = emotions.index(s) * np.ones([spectrum.shape[0], 1], np.float32)
            b = os.path.splitext(f)[0]
            _, b = os.path.split(b)
            spectrum = np.concatenate([spectrum, labels], 1)  # (frame_size, fft_size+1)
            np.save(join(save_path, '{}'.format(b)), spectrum)
            counter += 1
            data_list.append((join(s, '{}.npy'.format(b)), spectrum.shape[0]))
            total_frame += spectrum.shape[0]

    data_list.sort()
    print()
    with open(join(save_dir, mode, list_name), 'w') as f:
        for item in data_list:
            f.write("%s|%d\n" % (item[0], item[1]))
        # f.write("%d,%d\n" % (total_frame, spectrum.shape[1]))


def main_backup(emotions, load_dir):
    N = len(glob.glob(join(load_dir, '*', '*', '*.wav')))
    counter = 1
    train_list = []
    frame_list = []
    for d in list_dir(load_dir): # d : 'Training', 'Test'
        path = join(load_dir, d)
        for s in list_dir(path):
            path = join(load_dir, d, s)
            save_path = join(save_dir, d, s)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for f in list_full_filenames(path): # f:'ema00001.wav', ...
                print('\rFile {}/{}: {:50}'.format(counter, N, f), end='')
                spectrum = wav2spec(f)
                labels = emotions.index(s) * np.ones([spectrum.shape[0], 1], np.float32)
                b = os.path.splitext(f)[0]
                _, b = os.path.split(b)
                spectrum = np.concatenate([spectrum, labels], 1) # (frame_size, fft_size+1)
                np.save(join(save_path, '{}'.format(b)), spectrum)
                counter += 1
                train_list.append(join(s, '{}.npy'.format(b)))
                frame_list.append(spectrum.shape[0])

        print()
    print(frame_list)






def make_emotion_tsv(path):
    emotions = []
    for d in list_dir(path): # d = 'S_Neutral', 'T_Happy', ...
        emotions += list_dir(join(path, d))
    emotions = sorted(set(emotions))
    with open('./emotions.tsv', 'w') as f:
        for e in emotions:
            f.write('{}\n'.format(e))
    return emotions


def make_list(dir, list_name):
    list = []
    for d in list_dir(dir): # d = 'S_Neutral', 'T_Happy', ...
        path = join(dir ,d)
        files_list = os.listdir(path)
        for file_name in files_list:
            list.append(join(d, file_name))
    with open(join(dir, list_name), 'w') as f:
        for item in list:
            f.write("%s\n" % item)





if __name__ == '__main__':
    emotions = make_emotion_tsv(load_dir)
    main(emotions, 'Training', 'train.txt')
    main(emotions, 'Test', 'test.txt')
    #make_list(join(save_dir,'Training'), 'train.txt')
    #make_list(join(save_dir,'Test'), 'test.txt')
