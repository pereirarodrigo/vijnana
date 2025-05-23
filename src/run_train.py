import os
import yaml
import minigrid
from tqdm import tqdm
import gymnasium as gym
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import torch
from utils import train_policy
from modelling.reward_norm import IntrinsicRewardNormaliser
from modelling.policy import prepare_curiosity_module, build_actor_critic, build_policy_optim


# Load the config (YAML) file
with open("src/config.yml", 'r') as file:
    config = yaml.safe_load(file)


# Define the device
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)


def main() -> None:
    """
    Driver method to train the policy.
    """
    # Create the save path and folder
    full_path = f"{config['exp_args']['save_path']}/{config['exp_args']['env_id']}_pg_rl2"

    os.makedirs(full_path, exist_ok = True)

    # Create an env based on the config
    env = gym.make(id = config["exp_args"]["env_id"], render_mode = None)

    # Define and build the policy
    policy_module, critic_module, feature_ext = build_actor_critic(env, config, device)

    # Prepare the curiosity module
    rnd_module = prepare_curiosity_module(feature_ext, device)

    # Definite the loss, optimiser and LR scheduler
    optimiser, scheduler, combined_net_params = build_policy_optim(
        actor_head = policy_module, 
        critic_head = critic_module, 
        rnd_module = rnd_module, 
        shared_encoder = feature_ext,
        config = config
    )

    # Define a reward normaliser for intrinsic curiosity
    intrinsic_norm = IntrinsicRewardNormaliser(
        decay = config["curiosity_args"]["norm_decay"],
        eps = config["curiosity_args"]["norm_eps"],
        device = device
    )

    # Define logs and a progress bar
    logs = defaultdict(list)
    logger = SummaryWriter(log_dir = f"runs/{config['exp_args']['env_id']}_pg_rl2")
    progress_bar = tqdm(total = config["exp_args"]["improv_iters"])

    # Train the policy
    train_policy(
        env = env,
        policy_module = policy_module,
        critic_module = critic_module,
        rnd_module = rnd_module,
        feature_ext = feature_ext,
        intrinsic_norm = intrinsic_norm,
        optimiser = optimiser,
        scheduler = scheduler,
        combined_net_params = combined_net_params,
        logs = logs,
        tb_logger = logger,
        pbar = progress_bar,
        config = config,
        device = device,
        save_path = full_path
    )


if __name__ == "__main__":
    main()