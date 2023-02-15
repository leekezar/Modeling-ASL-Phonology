import torch.nn as nn


class Network(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class NNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x_enc = self.encoder(x)
        x_sign, x_params = self.decoder(x_enc)
        return x_sign, x_params #, x_enc
