import librosa
import numpy as np
import pysptk
from scipy.io import wavfile
import librosa.util
frame_length = 2048
hop_length = 512

order = 20

path = '../ETTS_newdata/data/wav/lmy00001.wav'
# LPC
sr, x = wavfile.read(path)
x = x.astype(np.float64)
librosa.util.valid_audio(x)
x = np.pad(x, int(frame_length // 2), mode='reflect')
frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
frames *= pysptk.blackman(frame_length)

lpc = pysptk.lpc(frames, order)
lpc[:, 0] = np.log(lpc[:, 0])


#MFCC


y, sr = librosa.load(path)
y = y.astype(np.float64)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
print(d)