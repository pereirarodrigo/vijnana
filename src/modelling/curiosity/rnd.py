import torch
import torch.nn as nn
from utils import layer_init

from typing import Tuple


class RND(nn.Module):
    """
    Implementation of random network distillation, as seen in Burda et al. (2018).
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Predictor model: predict next agent steps
        self.predictor_model = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 128)),
            nn.LayerNorm(128),
            nn.LeakyReLU(inplace = True),
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.LeakyReLU(inplace = True),
            layer_init(nn.Linear(128, 256))
        )

        # Set the predictor model to training mode
        self.predictor_model.train()

        # Target model: randomly initialised, sets the prediction problem
        self.target_model = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.Tanh()
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