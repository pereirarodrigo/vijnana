import gymnasium as gym

import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from modelling.network import layer_init, RND, RecurrentConvNet

from typing import List, Tuple


def prepare_curiosity_module(shared_encoder: RecurrentConvNet, device: str) -> nn.Module:
    """
    Create the RND curiosity module.
    """
    # Define the RND module
    rnd_module = RND(feature_dim = shared_encoder.output_dim).to(device)

    return rnd_module


def build_actor_critic(env: gym.Env, config: dict, device: str) -> Tuple[nn.Module, nn.Module, RecurrentConvNet]:
    """
    Create the actor-critic networks based on the provided environment.
    """
    # Extract environment parameters
    action_dim = env.action_space.n
    obs_shape = env.observation_space["image"].shape

    # Define the shared (CNN) module
    shared_encoder = RecurrentConvNet(
        input_shape = obs_shape,
        hidden_size = config["policy_args"]["num_cells"]
    ).to(device)

    # Define the actor and critic networks
    actor_head = layer_init(
        nn.Linear(config["policy_args"]["num_cells"], action_dim)
    ).to(device)

    critic_head = layer_init(
        nn.Linear(config["policy_args"]["num_cells"], 1)
    ).to(device)

    return actor_head, critic_head, shared_encoder


def build_policy_optim(
    actor_head: nn.Module,
    critic_head: nn.Module,
    rnd_module: RND,
    shared_encoder: nn.Module,
    config: dict
) -> Tuple[Optimizer, CosineAnnealingLR, List]:
    """
    Create the loss optimizer and learning rate scheduler for training.
    """
    # Combine parameters
    combined_params = list(actor_head.parameters()) + \
                      list(critic_head.parameters()) + \
                      list(shared_encoder.parameters()) + \
                      list(rnd_module.predictor_model.parameters())

    optim = Adam(combined_params, lr = config["policy_args"]["lr"], eps = 1e-5)

    scheduler = CosineAnnealingLR(
        optim, config["exp_args"]["num_steps"], 0.0
    )

    return optim, scheduler, combined_params