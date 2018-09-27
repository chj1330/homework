import os
import numpy as np
from Trainer import Trainer
#from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset
from os.path import join
import time
import torch
import argparse
from model import *
train_dir = './dataset/spectrogram/Training'
feat_dim = 1024 / 2 + 1 + 1
batch_size = 100
train_ratio = 0.8

class emotiondata(Dataset):
    def __init__(self, dir, list_name):
        self._X = []
        with open(join(dir, list_name)) as f:
            for line in f:
                fn = line.split()
                feature_path = join(dir, fn[0].split("|")[0])

                self._X.append(feature_path)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return np.load(self._X[idx])

def collate_fn(batch):
    lengths = [len(x) for x in batch]
    total_lengths = np.sum(lengths)
    emotion = [x[:,-1] for x in batch]
    spectrum = [x[:,:-1] for x in batch]
    data = {'Spectrum': spectrum, 'Emotion': emotion, 'Length' : lengths}
    return data, total_lengths



def main():
    GPU_USE = 1
    DEVICE = 'cuda:0'  # 0 : gpu0, 1 : gpu1, ...
    EPOCH = 20000
    BATCH_SIZE = 20
    LEARN_RATE = 0.002

    LOG_DIR = './log3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='log directory')
    parser.add_argument('--device', type=str, default=DEVICE, help='which device?')
    parser.add_argument('--gpu_use', type=int, default=GPU_USE, help='GPU enable? 0 : cpu, 1 : gpu')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='how many epoch?')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='how many batch?')
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE, help='learning rate')

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    train_data = emotiondata('./dataset/spectrogram/Training', 'train.txt')
    valid_data = emotiondata('./dataset/spectrogram/Training', 'valid.txt')

    train_loader = DataLoader(train_data, num_workers=8, collate_fn=collate_fn, batch_size=args.batch_size, drop_last=False, shuffle=True)
    valid_loader = DataLoader(valid_data, num_workers=8, collate_fn=collate_fn, batch_size=args.batch_size, drop_last=False, shuffle=True)

    if args.gpu_use == 1 and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.gpu_use == 0:
        device = torch.device('cpu')

    model = CCVAE2().to(device=device)
    trainer = Trainer(model=model, train_loader=train_loader, valid_loader=valid_loader, device=device, args=args)
    start_time = time.time()
    try:
        trainer.train()
        print("--- %s seconds spent ---" % (time.time() - start_time))

    except KeyboardInterrupt:
        print("--- %s seconds spent ---" %(time.time() - start_time))
        #trainer.save_checkpoint()
    #dset_train = SpectrumDataset(train_dir)
    #train_loader = DataLoader(dset_train, batch_size=5, shuffle=True, num_workers=8, drop_last=False)
    #sample = next(iter(train_loader))
    #spectrum = sample['Spectrum']



if __name__ == '__main__':
    main()

