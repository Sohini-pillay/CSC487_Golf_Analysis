import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision.models import mobilenet_v2

class EventDetector(nn.Module):
    def __init__(self, width_mult=1.0, lstm_layers=1, lstm_hidden=512, bidirectional=True, dropout=True, num_classes=9):

        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes

        # Load MobileNetV2 backbone
        net = mobilenet_v2(pretrained=True)

 
        self.cnn = net.features


        # Define LSTM
        self.rnn = nn.LSTM(
            input_size=int(1280 * width_mult if width_mult > 1.0 else 1280),
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

      
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, num_classes)
        else:
            self.lin = nn.Linear(self.lstm_hidden, num_classes)

    
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
    
        num_directions = 2 if self.bidirectional else 1
        return (
            Variable(torch.zeros(num_directions * self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
            Variable(torch.zeros(num_directions * self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True)
        )

    def forward(self, x):
        """
        Forward pass through EventDetector.
        """
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)  # Global average pooling
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in, self.hidden)

        # Classification
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, self.num_classes)

        return out
