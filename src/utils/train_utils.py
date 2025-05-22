import torch

from typing import Tuple


def compute_intrinsic_reward(predicted_phi_next: torch.Tensor, actual_phi_next: torch.Tensor) -> torch.Tensor:
    """
    Calculate intrinsic reward using L2 normalisation.
    """
    intrinsic_reward = 0.5 * (predicted_phi_next - actual_phi_next).pow(2).sum(dim = 1)

    return intrinsic_reward


def preprocess_obs(obs_dict, device: int ="cpu") -> torch.Tensor:
    """
    Re-arrange observation channel order.
    """
    img = obs_dict["image"]  # or "rgb"
    img = torch.tensor(img, dtype = torch.float32).to(device)

    return img.unsqueeze(0)
