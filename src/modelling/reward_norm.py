import torch


class IntrinsicRewardNormaliser:
    """
    Implementation of a normaliser for intrinsic rewards, useful to avoid agent
    overdependence on intrinsic signals and spiking gradients.
    """
    def __init__(self, decay: float = 0.99, eps: float = 1e-8, device: str = "cpu") -> None:
        self.decay = decay
        self.eps = eps
        self.device = device

        self.rnd_mean = torch.tensor(0.0, device = device)
        self.rnd_std = torch.tensor(1.0, device = device)


    def update(self, rnd_reward: torch.Tensor) -> None:
        """
        Update running stats from raw reward tensors.
        """
        self.rnd_mean = self.decay * self.rnd_mean + (1 - self.decay) * rnd_reward.mean()
        self.rnd_std = self.decay * self.rnd_std + (1 - self.decay) * rnd_reward.std(unbiased = False)


    def normalise(self, rnd_reward: torch.Tensor) -> torch.Tensor:
        """
        Apply normalisation using stored stats.
        """
        effective_std = (self.rnd_std + self.eps).clamp(min = 1e-4)
        norm_rnd = (rnd_reward - self.rnd_mean) / (effective_std + self.eps)

        return norm_rnd.clamp(-1.0, 1.0).unsqueeze(-1)