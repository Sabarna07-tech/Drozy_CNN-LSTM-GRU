import torch
import torch.nn as nn
from utils import normalizelayer

class Batchlayer(nn.Module):
    def __init__(self, dim):
        super(Batchlayer, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.gamma.data.uniform_(-0.1, 0.1)
        self.beta.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        data = normalizelayer(input)
        gammamatrix = self.gamma.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        betamatrix = self.beta.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        return data * gammamatrix + betamatrix

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.feature = 32
        self.padding = nn.ReplicationPad2d((31, 32, 0, 0))
        self.conv = nn.Conv2d(1, self.feature, (1, 64))
        self.batch = Batchlayer(self.feature)
        self.avgpool = nn.AvgPool2d((1, 8))
        self.fc = nn.Linear(32, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.lstm = nn.LSTM(32, 2)

    def forward(self, source):
        source = self.padding(source)
        source = self.conv(source)
        source = self.batch(source)
        source = nn.ELU()(source)
        source = self.avgpool(source)
        source = source.squeeze()
        source = source.permute(2, 0, 1)
        source = self.lstm(source)
        source = source[1][0].squeeze()
        source = self.softmax(source)
        return source