import os
import numpy as np
#from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset
from os.path import join
import time
import torch
import argparse
from torch.autograd import Variable
from model import CVAE2
train_dir = './dataset/spectrogram/Training'
feat_dim = 1024 / 2 + 1 + 1
batch_size = 100
train_ratio = 0.8

class checkdata(Dataset):
    """Face Landmarks dataset."""
    _X = []
    def __init__(self, dir):
        with open(dir+'/train.txt') as f:
            for line in f:
                self._X.append(join(dir, line.split()[0]))


    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        feature = np.load(self._X[idx])
        emotion = feature[:, -1] # last
        spectrum = feature[:, :-1] # except last

        sample = {'Spectrum': spectrum, 'Emotion': emotion}

        return sample

class emotiondata(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        entry = {'spectrum': self.x[index], 'emotion': self.y[index]}

        return entry

    def __len__(self):
        return self.x.shape[0]

class emotiondata2(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        entry = {'spectrum': self.x[index], 'emotion': self.y[index]}

        return entry

    def __len__(self):
        return self.x.shape[0]

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 513), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def load_data(dir):
    with open(dir + '/train.txt') as f:
        fns_tr = f.readlines() # list 'S_Neutral/ema00001.npy', 'S_Neutral/ema00002.npy', ...
    frame_size = int(fns_tr[-1].split(",")[0])
    fft_size = int(fns_tr[-1].split(",")[1]) - 1

    data_X = np.zeros((frame_size, fft_size), dtype=np.float32)
    data_y = np.zeros((frame_size, 1), dtype=np.float32)
    idx = 0
    for i in range(fns_tr.__len__() - 1):
        fn = fns_tr[i].split()
        feature = np.load(join(dir, fn[0].split("|")[0]))
        frm = int(fn[0].split("|")[1])
        data_X[idx:idx+frm,:] = feature[:,:-1]
        data_y[idx:idx+frm,0] = feature[:, -1]
        idx += frm

    num_train_size = int(frame_size * train_ratio)
    index_shuf = np.random.permutation(data_y.__len__())
    data_X = data_X[index_shuf]
    data_y = data_y[index_shuf]
    train_X = data_X[:num_train_size]
    train_y = data_y[:num_train_size]
    valid_X = data_X[num_train_size:]
    valid_y = data_y[num_train_size:]

    return train_X, train_y, valid_X, valid_y

def train(model, train_loader, valid_loader, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    num_epochs = args.epoch
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            X = data['spectrum']
            y = data['emotion']

            X = Variable(X).to(device)
            y = Variable(y).to(device)

            optimizer.zero_grad()
            recon_X, mu, logvar = model(X, y)
            loss = loss_function(recon_X, X, mu, logvar)
            loss.backward()
            optimizer.step()

            #if (i+1) % 10 == 0:
            #  print("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))

        print("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0]))
        eval_loss = eval(model, valid_loader, device)

        scheduler.step(eval_loss)
        curr_lr = optimizer.param_groups[0]['lr']

        if curr_lr < 1e-8:
            print("Early stopping\n\n")
            break


def eval(model, valid_loader, device):
    eval_loss = 0.0
    model.eval()
    for i, data in enumerate(valid_loader):
        X = data['spectrum']
        y = data['emotion']
        # have to convert to an autograd.Variable type in order to keep track of the gradient...
        X = Variable(X).to(device)
        y = Variable(y).to(device)

        recon_X, mu, logvar = model(X, y)
        loss = loss_function(recon_X, X, mu, logvar)

        eval_loss += loss.data[0]

    avg_loss = eval_loss / len(valid_loader)
    print('Average loss: {:.4f} \n'.format(avg_loss))

    return avg_loss


def main():
    GPU_USE = 1
    DEVICE = 'cuda:0'  # 0 : gpu0, 1 : gpu1, ...
    EPOCH = 20000
    BATCH_SIZE = 128
    LEARN_RATE = 0.01

    EXPORT_DIR = './checkpoint'
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, default=EXPORT_DIR, help='export directory')
    parser.add_argument('--device', type=str, default=DEVICE, help='which device?')
    parser.add_argument('--gpu_use', type=int, default=GPU_USE, help='GPU enable? 0 : cpu, 1 : gpu')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='how many epoch?')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='how many batch?')
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE, help='learning rate')

    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)
    train_X, train_y, valid_X, valid_y = load_data(train_dir)
    train_data = emotiondata(train_X, train_y)
    valid_data = emotiondata(valid_X, valid_y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False)

    if args.gpu_use == 1 and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.gpu_use == 0:
        device = torch.device('cpu')

    model = CVAE2().to(device=device)
    start_time = time.time()
    train(model, train_loader, valid_loader, device, args)
    print("--- %s seconds spent ---" %(time.time() - start_time))
    #dset_train = SpectrumDataset(train_dir)
    #train_loader = DataLoader(dset_train, batch_size=5, shuffle=True, num_workers=8, drop_last=False)
    #sample = next(iter(train_loader))
    #spectrum = sample['Spectrum']
    print(args.export_dir)


if __name__ == '__main__':
    main()

