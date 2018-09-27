from torch.autograd import Variable
import torch
from os.path import join
from tensorboardX import SummaryWriter
from analyzer import inv_spectrogram
from tqdm import tqdm
import numpy as np
from analyzer import get_emotions, save_wav, idx2onehot
class Trainer:
    def __init__(self, model, train_loader=None, valid_loader=None, device=torch.device('cpu'), args=None):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        if not train_loader == None:
            self.Writer = SummaryWriter(log_dir=args.log_dir)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learn_rate)

    def concat_batch(self, data):
        batch_length = data[1]
        X = data[0]['Spectrum']
        y = data[0]['Emotion']
        frm = data[0]['Length']
        data_X = np.zeros((batch_length, 513), dtype=np.float32)
        #data_y = np.zeros((batch_length, 1), dtype=np.float32)
        data_y = np.zeros((batch_length, 4), dtype=np.float32)
        idx = 0
        for i in range(X.__len__()):
            data_X[idx:idx + frm[i], :] = X[i]
            onehot_y = idx2onehot(y[i], 4)
            #data_y[idx:idx + frm[i], 0] = y[i]
            data_y[idx:idx + frm[i], :] = onehot_y
            idx += frm[i]
        return data_X, data_y

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 513), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD



    def train(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=3, verbose=True)
        self.model.train()
        for epoch in range(self.args.epoch):
            sum_loss = 0
            sum_len = 0
            for step, data in enumerate(tqdm(self.train_loader)):
                batch_X, batch_y = self.concat_batch(data)
                batch_X = Variable(torch.from_numpy(batch_X)).to(self.device)
                batch_y = Variable(torch.from_numpy(batch_y)).to(self.device)
                self.optimizer.zero_grad()
                recon_batch_X, mu, logvar = self.model(batch_X, batch_y)
                loss = self.loss_function(recon_batch_X, batch_X, mu, logvar)
                loss.backward()
                self.optimizer.step()
                sum_len += len(batch_X)
                sum_loss += loss.item()
            avg_loss = sum_loss / sum_len
            curr_lr = self.optimizer.param_groups[0]['lr']
            self.Writer.add_scalar("train loss (per epoch)", avg_loss, epoch)
            self.Writer.add_scalar("learning rate (per epoch)", curr_lr, epoch)
            print("Epoch [%d/%d], loss : %.4f" % (epoch + 1, self.args.epoch, avg_loss))
            if (epoch +1) % 10 == 0 :

                eval_loss = self.eval(epoch)
                scheduler.step(eval_loss)

                if curr_lr < 1e-8:
                    print("Early stopping\n\n")
                    break


    def eval(self, epoch):
        sum_loss = 0.0
        sum_len = 0
        self.model.eval()
        for step, data in enumerate(self.valid_loader):
            batch_X, batch_y = self.concat_batch(data)
            batch_X = Variable(torch.from_numpy(batch_X), volatile=False).to(self.device)
            batch_y = Variable(torch.from_numpy(batch_y), volatile=False).to(self.device)

            recon_batch_X, mu, logvar = self.model(batch_X, batch_y)
            loss = self.loss_function(recon_batch_X, batch_X, mu, logvar)

            sum_loss += loss.item()
            sum_len += len(batch_X)

            #if epoch % 1000 == 0:
            #spectrum_X = data[0]['Spectrum']
            #emotion_X = data[0]['Emotion']
            #for idx, val in enumerate(emotion_X):
            #    dd = max(val==0)
            #    print(dd)
            #print(spectrum_X)

        if (epoch + 1) % 1000 == 0:
            self.save_checkpoint(epoch)
        avg_loss = sum_loss / sum_len
        self.Writer.add_scalar("eval loss (per epoch)", avg_loss, epoch)
        print('Average eval loss: {:.4f} \n'.format(avg_loss))

        return avg_loss

    def test(self, source_X, spectrum_path=None, wav_path=None):
        self.model.eval()

        source_X = Variable(torch.from_numpy(source_X), volatile=False).to(self.device)
        #source_y = Variable(torch.from_numpy(source_y), volatile=False).to(self.device)
        emotions = get_emotions()

        for emo in emotions :
            target_y = emotions.index(emo) * np.ones([source_X.shape[0], 1], np.float32)
            target_y = idx2onehot(target_y, 4)
            target_y = Variable(torch.from_numpy(target_y), volatile=False).to(self.device)
            recon_target_X, mu, logvar = self.model(source_X, target_y)
            recon_target_X = torch.t(recon_target_X)
            recon_target_X = recon_target_X.data.cpu().numpy()
            target_X_wav = inv_spectrogram(recon_target_X)
            emotion_spectrum_path = spectrum_path.replace('.npy', '-{}.npy'.format(emo))
            np.save(emotion_spectrum_path, recon_target_X, allow_pickle=False)
            emotion_wav_path = wav_path.replace('.wav', '-{}.wav'.format(emo))
            save_wav(target_X_wav, emotion_wav_path)


        #recon_source_X, mu, logvar = self.model(source_X, source_y)

        #recon_source_X = torch.t(recon_source_X)

        #comparison = torch.cat([torch.t(t_X), recon_t_X])
        #save_image(recon_t_X.data.cpu(), 'results_s2t/reconstruction_' + str(epoch) + '.png')
        #save_image(recon_s_X.data.cpu(), 'results_s2s/reconstruction_' + str(epoch) + '.png')
        #source_X_wav = inv_spectrogram(recon_source_X.data.cpu().numpy())





    def save_checkpoint(self, epoch):

        checkpoint_path = join(
            self.args.log_dir, "checkpoint_epoch{:09d}.pth".format(epoch+1))
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)