import torch
import torch.nn as nn
from utils import layer_init

from typing import Tuple


class RecurrentConvNet(nn.Module):
    """
    Implementation of a recurrent convolutional neural network (CNN) for feature extraction.
    """
    def __init__(self, input_shape: tuple[int, int, int], hidden_size: int = 128) -> None:
        super().__init__()
        channels, height, width = input_shape

        # Convolutional block
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(channels, 16, kernel_size = 3, stride = 1, padding = 1)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out_size = self.conv(dummy).shape[-1]
        
        self.rnn = nn.GRU(input_size = conv_out_size, hidden_size = hidden_size, batch_first = True)
        self.output_dim = hidden_size


    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass in the network.
        """
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.conv(x)
        x = x.view(B, T, -1)
        out, h = self.rnn(x, h)

        return out, h
