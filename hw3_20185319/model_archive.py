'''
model_archive.py

A file that contains neural network models.
You can also make different model like CNN if you follow similar format like given RNN.
'''
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CRNN, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,1), stride=(2,1)),
            nn.Dropout(0.5)
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1), stride=(2,1)),
            nn.Dropout(0.3)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((3,1))
            )
        self.lstm = nn.LSTM(32, 64, 1, bidirectional=True, dropout=0.25)
        self.linear = nn.Linear(2*64, num_classes)  # 2 for bidirection


    def forward(self, x):
        # seq_len, batch, input_size = 15, 128, 12
        x = x.permute(1, 2, 0)
        x = x.contiguous()
        # batch, input_size, seq_len = 128, 12, 15
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        # batch, in_channel, input_size, seq_len = 128, 1, 12, 20
        output = self.conv0(x)
        # batch, in_channel, input_size, seq_len = 128, 32, 6, 20
        output = self.conv1(output)
        # batch, in_channel, input_size, seq_len = 128, 64, 3, 20
        output = self.conv2(output)
        # batch, in_channel, input_size, seq_len = 128, 128, 3, 20
        output = self.conv3(output)
        # batch, in_channel, input_size, seq_len = 128, 32, 1, 20
        output = output.squeeze()
        # batch, in_channel, seq_len = 128, 32, 20
        output = output.permute(2, 0, 1)
        # batch, in_channel, seq_len = 20, 128, 32
        # 20, 128, 32
        output, _ = self.lstm(output, None)
        # 20, 128, 128
        output = self.linear(output[-1])
        # 128, 25
        output = nn.functional.log_softmax(output, dim=1)
        return output
