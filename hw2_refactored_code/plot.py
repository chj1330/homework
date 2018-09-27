import numpy as np
from matplotlib import pyplot as plt
import librosa
file_name = '/home/chj/project/2018/chj/gct634-2018/hw2/gtzan/blues/blues.00000.au'
y,sr = librosa.load(file_name,sr=22050)
S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)
X = librosa.amplitude_to_db(S, ref=np.max)

mel = np.load('/home/chj/project/2018/chj/gct634-2018/hw2/gtzan_mel/blues/blues.00000.npy')
"""
mel_u = np.load('/home/chj/project/2018/chj/gct634-2018/hw2/gtzan_mel_pitch_u/blues/blues.00000.npy')
mel_d = np.load('/home/chj/project/2018/chj/gct634-2018/hw2/gtzan_mel_pitch_d/blues/blues.00000.npy')
plt.subplot(311)
plt.pcolormesh(mel)
plt.subplot(312)
plt.pcolormesh(mel_u)
plt.subplot(313)
plt.pcolormesh(mel_d)
plt.show()
"""
plt.subplot(311)
plt.plot(y)
plt.subplot(312)
plt.pcolormesh(X)
plt.subplot(313)
plt.pcolormesh(mel)
plt.show()