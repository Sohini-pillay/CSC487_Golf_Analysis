import torch
import torch.nn as nn
import math
from torchvision.models import mobilenet_v2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if d_model is odd, pad one column to match dimensions
            pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EventDetector(nn.Module):
    def __init__(self, width_mult=1.0, num_transformer_layers=2,
                 num_heads=8, transformer_dim=1280, dropout=True, num_classes=9):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.num_classes = num_classes
        self.dropout = dropout

        # Load MobileNetV2 backbone
        net = mobilenet_v2(pretrained=True)
        self.cnn = net.features

        # transformer setup
        # transformer dimensions need to match output of CNN
        self.pos_encoder = PositionalEncoding(d_model=transformer_dim, dropout=0.5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dropout=0.5)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # classification layer for which key frame
        self.lin = nn.Linear(transformer_dim, num_classes)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        """x: tensor of shape (batch_size, timesteps, C, H, W)"""
        batch_size, timesteps, C, H, W = x.size()

        # CNN forward: process each frame independently
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)  # global average pooling, shape: (batch_size * timesteps, transformer_dim)
        if self.dropout:
            c_out = self.drop(c_out)

        # reshape for transformer: (batch_size, timesteps, transformer_dim)
        r_in = c_out.view(batch_size, timesteps, -1)
        # apply positional encoding
        r_in = self.pos_encoder(r_in)
        # transpose to (timesteps, batch_size, transformer_dim) as required by the transformer
        r_in = r_in.transpose(0, 1)
        # transformer forward pass
        r_out = self.transformer(r_in)
        # transpose back to (batch_size, timesteps, transformer_dim)
        r_out = r_out.transpose(0, 1)

        # classification (apply linear layer to each timestep)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, self.num_classes)

        return out
