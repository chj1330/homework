from os.path import join
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch
train_dir = './dataset/spectrogram/Training'
"""
class checkdata(Dataset):

    _X = []
    def __init__(self, dir):
        with open(dir) as f:
            for line in f:
                fn = line.split()
                feature_path = join(dir, fn[0].split("|")[0])
                self._X.append(feature_path)


    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        feature = np.load(self._X[idx])
        emotion = feature[:, -1] # last
        spectrum = feature[:, :-1] # except last

        sample = {'Spectrum': spectrum, 'Emotion': emotion}

        return sample

class checkdata2(Dataset):

    def __init__(self, dir, list_name):
        self._X = []
        with open(join(dir,list_name)) as f:
            for line in f:
                fn = line.split()
                feature_path = join(dir, fn[0].split("|")[0])

                self._X.append(feature_path)



    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        feature = np.load(self._X[idx])
        #emotion = feature[:, -1] # last
        #spectrum = feature[:, :-1] # except last
        #frame_len = self._frame_len[idx]
        #sample = {'Spectrum': spectrum, 'Emotion': emotion, "Length" : frame_len}

        return feature

def collate_fn(batch):
    lengths = [len(x) for x in batch]
    total_lengths = np.sum(lengths)
    emotion = [x[:,-1] for x in batch]
    spectrum = [x[:,:-1] for x in batch]
    data = {'Spectrum': spectrum, 'Emotion': emotion, 'Length' : lengths}
    print(total_lengths)
    return data, total_lengths



def concat_batch(data):
    batch_length = data[1]
    X = data[0]['Spectrum']
    y = data[0]['Emotion']
    frm = data[0]['Length']
    data_X = np.zeros((batch_length, 513), dtype=np.float32)
    data_y = np.zeros((batch_length, 1), dtype=np.float32)
    idx = 0
    for i in range(X.__len__()):
        data_X[idx:idx+frm[i],:] = X[i]
        data_y[idx:idx+frm[i],0] = y[i]
        idx += frm[i]

    return data_X, data_y



    #sample = {'Spectrum': spectrum, 'Emotion': emotion}
train_ratio = 0.8


train_data = checkdata2('./dataset/spectrogram/Training', 'train.txt')
valid_data = checkdata2('./dataset/spectrogram/Training', 'valid.txt')

train_loader = DataLoader(train_data, num_workers=8, collate_fn=collate_fn, batch_size=4)
valid_loader = DataLoader(valid_data, num_workers=8, collate_fn=collate_fn, batch_size=2)

for i, data in enumerate(train_loader):
    print(data)
    batch_X, batch_y = concat_batch(data)
    device = 'cuda:0'

    batch_X = Variable(torch.from_numpy(batch_X)).to(device)
    batch_y = Variable(torch.from_numpy(batch_y)).to(device)
"""


"""
z_range = 2.0
n_img_y = 4
n_img_x = 11
v = z_range * 0.7
z = [[v, v], [-v, v], [v, -v], [-v, -v]]
z2 = 1.4 * np.ones((513,1))
repeat_shape = list(np.int32(np.ones(n_img_y) * n_img_x))
z = np.repeat(z, repeat_shape, axis=0)
z = np.clip(z, -z_range, z_range)
fake_id_PARR = np.zeros(shape=[z.shape[0], 10])
for i in range(z.shape[0]):
    if i % 11 == 0:  # template
        label = 3  # let's fix label for template as 3 for better style-comparison.
    else:
        label = (i % 11) - 1
    fake_id_PARR[i, label] = 1.0
print(z)
"""
n=100
neu = np.load('result2/ema00096-S_Neutral.npy').T[n]
ang = np.load('result2/ema00096-T_Angry.npy').T[n]
hap = np.load('result2/ema00096-T_Happy.npy').T[n]
sad = np.load('result2/ema00096-T_Sad.npy').T[n]

import matplotlib.pyplot as plt
plt.subplot(411)
plt.plot(neu)
plt.subplot(412)
plt.plot(ang)
plt.subplot(413)
plt.plot(hap)
plt.subplot(414)
plt.plot(sad)
plt.show()

dd = neu - ang

print(dd)
