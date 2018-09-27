import os
import numpy as np
from Trainer import Trainer
from torch.utils.data import DataLoader, Dataset
from os.path import join
import time
import argparse
import torch
from model import CCVAE2

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
    GPU_USE = 0
    DEVICE = 'cuda:1'  # 0 : gpu0, 1 : gpu1, ...

    TEST_DIR = './dataset/spectrogram/Test/S_Neutral'
    RESULT_DIR = './result2'
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dir', type=str, default=TEST_DIR, help='log directory')
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR, help='log directory')
    parser.add_argument('--device', type=str, default=DEVICE, help='which device?')
    parser.add_argument('--gpu_use', type=int, default=GPU_USE, help='GPU enable? 0 : cpu, 1 : gpu')


    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    if args.gpu_use == 1 and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.gpu_use == 0:
        device = torch.device('cpu')

    model = CCVAE2().to(device=device)
    checkpoint = torch.load('log3/checkpoint_epoch000002000.pth')
    model.load_state_dict(checkpoint['state_dict'])

    trainer = Trainer(model=model, device=device, args=args)
    with open(join(args.test_dir, 'train.txt'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            spectrum_path = join(args.result_dir, parts[0])
            wav_path = join(args.result_dir, parts[0].replace('.npy', '.wav'))
            source = np.load(join(args.test_dir, parts[0]))
            source_X = source[:,:-1]
            start_time = time.time()
            trainer.test(source_X, spectrum_path, wav_path)
            print("--- %s seconds spent ---" % (time.time() - start_time))
            print('%s' % parts[0])





        #trainer.save_checkpoint()
    #dset_train = SpectrumDataset(train_dir)
    #train_loader = DataLoader(dset_train, batch_size=5, shuffle=True, num_workers=8, drop_last=False)
    #sample = next(iter(train_loader))
    #spectrum = sample['Spectrum']



if __name__ == '__main__':
    main()

