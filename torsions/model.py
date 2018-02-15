from torch import nn, cat
import torch.nn.functional as F

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv1d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(20, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv1d(64, 6, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        u = F.upsample(enc4, center.size()[2:], mode='linear')
        dec4 = self.dec4(cat([center, F.upsample(enc4, center.size()[2:], mode='linear')], 1))
        dec3 = self.dec3(cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='linear')], 1))
        dec2 = self.dec2(cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='linear')], 1))
        dec1 = self.dec1(cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='linear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='linear')

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


from torch import zeros
from torch.autograd import Variable
from torch import cuda

class LSTMaa(nn.Module):

    def __init__(self):
        super(LSTMaa, self).__init__()

        self.hidden_dim = 64

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(20, self.hidden_dim // 2, bidirectional=True, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2target = nn.Linear(64, 6)

        # print(self.hidden)

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if cuda.is_available():
            return (Variable(zeros(2, minibatch_size, self.hidden_dim // 2)).cuda(),
                    Variable(zeros(2, minibatch_size, self.hidden_dim // 2)).cuda())
        else:
            return (Variable(zeros(2, minibatch_size, self.hidden_dim // 2)),
                    Variable(zeros(2, minibatch_size, self.hidden_dim // 2)))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)

        target = self.hidden2target(lstm_out)
        return target