# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import torch
import torch.nn as nn

# model class


class model_2DCNN(nn.Module):
    def __init__(self):
        super(model_2DCNN, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(8, stride= 8, padding=(1,0)))

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((6, 8), stride=2))

        self.fc0 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(64, 10)
        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        out = self.conv0(x)
        out = self.conv1(out)

        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2)* out.size(3))
        out = self.fc0(out)

        out = nn.functional.log_softmax(out, dim=1)

        return out


