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
class model_1DCNN(nn.Module):
    def __init__(self):
        super(model_1DCNN, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=2),
            nn.Dropout(0.1))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=2),
            nn.Dropout(0.2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.3))

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(3, stride=3),
            nn.Dropout(0.5))

        self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 10)
        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128

        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2))
        out = self.fc0(out)
        out = self.fc1(out)
        #out = nn.functional.log_softmax(out, dim=1)

        return out


class model_1DCNN_2(nn.Module):
    def __init__(self):
        super(model_1DCNN_2, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.2))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(8, stride=2),
            nn.Dropout(0.5))

        self.fc0 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(64, 10)
        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128

        out = self.conv0(x)
        out = self.conv1(out)

        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2))
        out = self.fc0(out)

        out = nn.functional.log_softmax(out, dim=1)

        return out

class model_1DCNN_2_dx(nn.Module):
    def __init__(self):
        super(model_1DCNN_2_dx, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(8, stride=2))

        self.fc0 = nn.Linear(128, 10)

        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128

        out = self.conv0(x)
        out = self.conv1(out)

        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2))
        out = self.fc0(out)

        out = nn.functional.log_softmax(out, dim=1)

        return out



class model_1DCNN_3(nn.Module):
    def __init__(self):
        super(model_1DCNN_3, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(0.1))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(0.2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.avgpool = nn.AvgPool1d(3, stride=3)
        self.maxpool = nn.MaxPool1d(3, stride=3)
        self.dropout = nn.Dropout(0.4)

        self.fc0 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 10)
        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128

        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out1 = self.avgpool(out)
        out2 = self.maxpool(out)
        out = torch.cat((out1, out2), dim=2)
        out = self.dropout(out)
        out = out.view(x.size(0), out.size(1) * out.size(2))
        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here

        out = self.fc0(out)
        out = self.fc1(out)

        out = nn.functional.log_softmax(out, dim=1)

        return out

class model_1DCNN_3_dx(nn.Module):
    def __init__(self):
        super(model_1DCNN_3_dx, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4))

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.avgpool = nn.AvgPool1d(3, stride=3)
        self.maxpool = nn.MaxPool1d(3, stride=3)

        self.fc0 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 10)
        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128

        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out1 = self.avgpool(out)
        out2 = self.maxpool(out)
        out = torch.cat((out1, out2), dim=2)
        #out = self.dropout(out)
        out = out.view(x.size(0), out.size(1) * out.size(2))
        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here

        out = self.fc0(out)
        out = self.fc1(out)

        out = nn.functional.log_softmax(out, dim=1)

        return out

class model_1DCNN_4(nn.Module):
    def __init__(self):
        super(model_1DCNN_4, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(8, stride=8),
            nn.Dropout(0.2))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool2d(4, stride=2),
            nn.Dropout(0.5))

        self.fc0 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(64, 10)
        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128

        out = self.conv0(x)
        out = self.conv1(out)

        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2))
        out = self.fc0(out)

        out = nn.functional.log_softmax(out, dim=1)

        return out

class model_2DCNN(nn.Module):
    def __init__(self):
        super(model_2DCNN, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(8, stride= 8, padding=(1,0)),
            nn.Dropout(0.2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((6, 8), stride=2),
            nn.Dropout(0.5))

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

class model_2DCNN_dx(nn.Module):
    def __init__(self):
        super(model_2DCNN_dx, self).__init__()

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

class model_predict(nn.Module):
    def __init__(self):
        super(model_predict, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(9, 1),
            nn.Dropout(0.1))
        #self.fc0 = nn.Linear(9, 1)
        # self.activation = nn.Softmax()

    def forward(self, x):
        # input x: minibatch x 128 x 12XX

        # print(x.size())
        # x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 10 x 10
        out = self.linear(x)
        out = out.view(out.size(0), out.size(1))
        return out


