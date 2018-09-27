from dtw import dtw
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from librosa.display import specshow
from analyzer import inv_spectrogram, save_wav
from os.path import join
# frame, fft
def neu2emo(fn, emo, num):
    e_fn = fn.replace('S_Neutral', emo)
    e_fn = e_fn.replace('ema000', 'ema00{}'.format(num))
    return e_fn
"""
with open('dataset/spectrogram/Test/S_Neutral/train.txt') as f:
    for line in f:
        fn = line.split()
        n_fn = fn[0].split("|")[0]
        neu_path = join('dataset/spectrogram/Test/S_Neutral', n_fn)
        neu = np.load(neu_path)
        neu = neu[:,:-1]
        h_fn = neu2emo(n_fn, 'T_Happy', 1)
        hap_path = join('dataset/spectrogram/Test/T_Happy', h_fn)
        hap = np.load(hap_path)
        hap = hap[:,:-1]
        dist, cost, acc, path = dtw(neu, hap, dist=lambda neu, hap: norm(neu - hap, ord=1))
        dtw_neu = np.asarray([neu[path[0][i]] for i in range(path[0].size)])
        dtw_hap = np.asarray([hap[path[1][i]] for i in range(path[1].size)])
        np.save(hap_path, dtw_hap)
        np.save(neu_path, dtw_neu)
        print(fn)
"""


neutral = np.load('dataset/spectrogram/Test/S_Neutral/ema00100.npy')
neutral = neutral[:,:-1]

#happy = np.load('dataset/spectrogram/Training/T_Happy/ema00101.npy')
#happy = happy[:,:-1]
sad = np.load('dataset/spectrogram/Test/T_Happy/ema00200.npy')
sad = sad[:,:-1]
dist, cost, acc, path = dtw(neutral, sad, dist=lambda neutral, happy: norm(neutral - happy, ord=1))

dtw_happy = np.asarray([neutral[path[0][i]] for i in range(path[0].size)])
dtw_sad = np.asarray([sad[path[1][i]] for i in range(path[1].size)])

np.save('dataset/spectrogram/Test/S_Neutral/ema00100.npy', dtw_happy)
np.save('dataset/spectrogram/Test/T_Happy/ema00200.npy', dtw_sad)
"""
plt.subplot(211)
specshow(dtw_happy.T)
plt.subplot(212)
specshow(dtw_sad.T)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

dtw_sad_wav = inv_spectrogram(dtw_sad.T)
dtw_happy_wav = inv_spectrogram(dtw_happy.T)
save_wav(dtw_sad_wav, 'dtw_sad.wav')
save_wav(dtw_happy_wav, 'dtw_neutral2.wav')
"""

