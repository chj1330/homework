import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

class CVAE_con(nn.Module):
    def __init__(self):
        super(CVAE_con, self).__init__()

        self.fc1 = nn.Linear(513, 513)
        self.fc21 = nn.Linear(513, 512)
        self.fc22 = nn.Linear(513, 512)
        self.fc3 = nn.Linear(513, 513)
        self.fc4 = nn.Linear(513, 513)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        x2 = x.view(-1, 513)
        mu, logvar = self.encode(x.view(-1, 513))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar




class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(513, 513)
        self.fc21 = nn.Linear(513, 512)
        self.fc22 = nn.Linear(513, 512)
        self.fc3 = nn.Linear(513, 513)
        self.fc4 = nn.Linear(513, 513)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        # frame_size, fft_size
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

class CCVAE(nn.Module):
    def __init__(self):
        super(CCVAE, self).__init__()
        # Encoder
        self.en1 = nn.Conv1d(1, 16, kernel_size=9, stride=3, bias=False)
        self.en2 = nn.Conv1d(16, 32, kernel_size=7, stride=3, bias=False)
        self.en3 = nn.Conv1d(32, 64, kernel_size=7, stride=3, bias=False)
        self.fc1 = nn.Linear(1088, 512)
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)
        # Decoder
        self.fc3 = nn.Linear(513, 513)
        self.fc4 = nn.Linear(513, 1088)
        self.de1 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=3, bias=False)
        self.de2 = nn.ConvTranspose1d(32, 16, kernel_size=7, stride=3, bias=False)
        self.de3 = nn.ConvTranspose1d(16, 1, kernel_size=9, stride=3, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.en1(x))
        # h1 = batch, 16, 169
        h = self.relu(self.en2(h))
        # h2 = batch, 16, 55
        h = self.relu(self.en3(h))
        # h3 = batch, 64, 17
        h = h.view(h.size(0), h.size(1)*h.size(2))
        # h3 = batch, 1088
        h = self.relu(self.fc1(h))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        # inputs = batch, 512+1=513
        h2 = self.relu(self.fc3(inputs))
        h2 = self.relu(self.fc4(h2))
        h2 = h2.view(h2.size(0), 64, 17)
        # h2 = batch, 64, 17
        h2 = self.relu(self.de1(h2))
        # h2 = batch, 32, 55
        h2 = self.relu(self.de2(h2))
        # h2 = batch, 16, 169
        h2 = self.de3(h2)
        h2 = h2.squeeze()
        # h2 = batch, 1, 513
        return self.sigmoid(h2)

    def forward(self, x, c):
        # batch, 1, fft_size
        x = x.view(x.size(0), 1, x.size(1))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class CVAE2(nn.Module):
    def __init__(self):
        super(CVAE2, self).__init__()

        self.fc1 = nn.Linear(514, 513)
        self.fc21 = nn.Linear(513, 512)
        self.fc22 = nn.Linear(513, 512)
        self.fc3 = nn.Linear(513, 513)
        self.fc4 = nn.Linear(513, 513)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        # frame_size, fft_size
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(513, 277)
        self.fc21 = nn.Linear(277, 40)
        self.fc22 = nn.Linear(277, 40)
        self.fc3 = nn.Linear(40, 277)
        self.fc4 = nn.Linear(277, 513)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 513))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar