import torch
import numpy as np
import torch.nn as nn

from typing import Tuple


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Initialise a neural network's layers' weight and bias parameters.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


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


class RND(nn.Module):
    """
    Implementation of random network distillation, as seen in Burda et al. (2018).
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Predictor model: predict next agent steps
        self.predictor_model = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.LeakyReLU(inplace = True),
            layer_init(nn.Linear(256, 256)),
            nn.LeakyReLU(inplace = True),
            layer_init(nn.Linear(256, 256))
        )

        # Set the predictor model to training mode
        self.predictor_model.train()

        # Target model: randomly initialised, sets the prediction problem
        self.target_model = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256))
        )

        # Target network is not trainable, so lock it out of training
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.target_model.eval()


    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass in the network.
        """
        with torch.no_grad():
            target = self.target_model(features)

        pred = self.predictor_model(features)
        
        return target, pred