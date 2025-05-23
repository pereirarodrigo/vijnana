import torch
import numpy as np
import torch.nn as nn


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Initialise a neural network's layers' weight and bias parameters.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


def compute_intrinsic_reward(predicted_phi_next: torch.Tensor, actual_phi_next: torch.Tensor) -> torch.Tensor:
    """
    Calculate intrinsic reward using L2 normalisation.
    """
    intrinsic_reward = 0.5 * (predicted_phi_next - actual_phi_next).pow(2).sum(dim = 1)

    return intrinsic_reward


def preprocess_obs(obs_dict, device: str = "cpu") -> torch.Tensor:
    """
    Re-arrange observation channel order.
    """
    img = obs_dict["image"]
    img = torch.tensor(img, dtype = torch.float32).to(device)

    return img.unsqueeze(0)
