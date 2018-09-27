from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import librosa
import numpy as np
import lws
import os
from scipy.io import wavfile
from os.path import dirname, join, basename, splitext
from model import *
import dtw
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
fft_size = 1024
hop_size = 256
min_level_db = -100
ref_level_db = 20
power = 1.4
interval = 10000
preemphasis_factor = 0.97
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 513), size_average=False)
    #MSELoss = nn.MSELoss(size_average=False)

    #MSE = MSELoss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    data_X = Variable(spectrum)
    data_y = Variable(emotion)
    if args.cuda:
        data_X = data_X.cuda()
        data_y = data_y.cuda()
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data_X, data_y)
    loss = loss_function(recon_batch, data_X, mu, logvar)
    loss.backward()
    train_loss += loss.data[0]
    optimizer.step()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, len(data_X), len(data_X), 100., loss.data[0] / len(data_X)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))


def test(epoch):
    model.eval()
    test_loss = 0
    s_X = spectrum_s
    s_y = emotion_s
    t_y = emotion_s2t

    if args.cuda:
        s_X = s_X.cuda()
        s_y = s_y.cuda()
        t_y = t_y.cuda()
    s_X = Variable(s_X, volatile=True)
    t_y = Variable(t_y, volatile=True)
    s_y = Variable(s_y, volatile=True)

    recon_t_X, _, _ = model(s_X, t_y)
    recon_s_X, _, _ = model(s_X, s_y)
    #test_loss += loss_function(recon_batch, data_, mu, logvar).data[0]
    #print('====> Test set loss: {:.4f}'.format(test_loss))

    if epoch % interval == 0 :
        recon_t_X = torch.t(recon_t_X)
        recon_s_X = torch.t(recon_s_X)
        #comparison = torch.cat([torch.t(t_X), recon_t_X])
        save_image(recon_t_X.data.cpu(), 'results_s2t/reconstruction_' + str(epoch) + '.png')
        save_image(recon_s_X.data.cpu(), 'results_s2s/reconstruction_' + str(epoch) + '.png')
        recon_t_X_wav = inv_spectrogram(recon_t_X.data.cpu().numpy())
        wav_path = join('results_s2t', 'reconstruction_%s.wav' % str(epoch))
        save_wav(recon_t_X_wav, wav_path)
        recon_s_X_wav = inv_spectrogram(recon_s_X.data.cpu().numpy())
        wav_path_s = join('results_s2s', 'reconstruction_%s.wav' % str(epoch))
        save_wav(recon_s_X_wav, wav_path_s)


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

def GL_test():
    wav, _ = librosa.load('ema00001.wav', 22050)
    wav = wav / np.abs(wav).max() * 0.999
    spectrum = spectrogram(wav).astype(np.float32).T  # fft x frame -> frame(batch) x fft
    recon_wav = inv_spectrogram(spectrum.T)
    save_wav(recon_wav, 'GL_test.wav')


GL_test()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
wav, _ = librosa.load('ema00001.wav', 22050)
wav = wav / np.abs(wav).max() * 0.999
spectrum = spectrogram(wav).astype(np.float32).T # fft x frame -> frame(batch) x fft
spectrum_s = torch.from_numpy(spectrum)
emotion_s = np.zeros((spectrum_s.shape[0], 1)).astype('float32')
emotion_s = torch.from_numpy(emotion_s)

wav_t, _ = librosa.load('ema00101.wav', 22050)
wav_t = wav_t / np.abs(wav_t).max() * 0.999
spectrum_t = spectrogram(wav_t).astype(np.float32).T # fft x frame -> frame(batch) x fft

spectrum_t = torch.from_numpy(spectrum_t)
save_image(torch.t(spectrum_t), 'results_s2t/target.png')
emotion_t = np.ones((spectrum_t.shape[0], 1)).astype('float32')
emotion_t = torch.from_numpy(emotion_t)
spectrum = torch.cat((spectrum_s, spectrum_t), 0)
emotion = torch.cat((emotion_s, emotion_t), 0)



emotion_s2t = np.ones((spectrum_s.shape[0], 1)).astype('float32')
emotion_s2t = torch.from_numpy(emotion_s2t)
model = CCVAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



save_image(spectrum_t, 'results_s2t/target.png')

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    """
    if epoch % interval == 0 :
        sample = Variable(torch.randn(646, 40))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample)
        sample = torch.t(sample)
        save_image(sample.data.cpu(), 'results/sample_' + str(epoch) + '.png')
        recon_batch_wav = inv_spectrogram(sample.data.cpu().numpy())
        wav_path = join('results', 'sample_%s.wav' % str(epoch))
        save_wav(recon_batch_wav, wav_path)
    """




