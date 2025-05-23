import gc
import random
from tqdm import tqdm
from collections import defaultdict

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler   
from modelling.meta_learning import generate_meta_episode
from modelling.reward_norm import IntrinsicRewardNormaliser

from collections import deque


def train_policy(
    env: gym.Env,
    policy_module: nn.Module,
    critic_module: nn.Module,
    feature_ext: nn.Module,
    rnd_module: nn.Module,
    intrinsic_norm: IntrinsicRewardNormaliser,
    optimiser: Optimizer,
    scheduler: LRScheduler,
    combined_net_params: list,
    logs: defaultdict,
    tb_logger: SummaryWriter,
    pbar: tqdm,
    config: dict,
    device: str,
    save_path: str
) -> None:
    """
    Train the policy based on the provided parameters.
    """
    global_step = 0
    num_iters = config["exp_args"]["num_steps"]
    rollout_len = config["exp_args"]["rollout_steps"]
    improv_iters = config["exp_args"]["improv_iters"]
    total_meta_episodes = config["exp_args"]["total_meta_episodes"]
    num_pol_updates = config["exp_args"]["num_pol_updates"]
    meta_ep_per_pol_update = config["exp_args"]["meta_ep_per_pol_upd"]

    # Deque for meta-learning episode returns
    meta_ep_returns = deque(maxlen = 1000)

    for improv_idx in range(improv_iters):
        # Collect meta-episodes
        meta_episodes = []
        total_rnd_losses, total_raw_rnds, total_norm_rnds, total_extr_rews = [], [], [], []

        # Collect meta-episodes and update the policy
        for _ in range(total_meta_episodes):
            # Generate meta-episodes and collect metrics
            ep_obs, ep_rewards, ep_actions, ep_ext_rewards, rnd_losses, raw_rnds, norm_rnds = generate_meta_episode(
                env = env,
                policy_module = policy_module,
                critic_module = critic_module,
                feature_ext = feature_ext,
                rnd_module = rnd_module,
                intrinsic_norm = intrinsic_norm,
                episode_len = num_iters,
                rollout_len = rollout_len,
                device = device
            )

            # Append to the list of meta-episodes
            meta_episodes.append([
                ep_obs.cpu(),
                ep_rewards.cpu(),
                ep_actions.cpu()
            ])

            # Append RND results
            total_rnd_losses.extend(rnd_losses)
            total_raw_rnds.extend(raw_rnds)
            total_norm_rnds.extend(norm_rnds)
            
            # Compute episode return
            total_ep_rew = torch.sum(ep_rewards).item()
            total_ext_rew = torch.sum(ep_ext_rewards).item()
            
            total_extr_rews.append(total_ext_rew)
            meta_ep_returns.append(total_ep_rew)

        # Update the policy
        for _ in range(num_pol_updates):
            # Sample batch of meta-episodes
            sampled = random.sample(meta_episodes, k = meta_ep_per_pol_update)

            # Concatenate batch into tensors
            meta_obs = torch.cat([e[0].to(device) for e in sampled], dim = 0)
            meta_rew = torch.cat([e[1].to(device) for e in sampled], dim = 0)
            meta_act = torch.cat([e[2].to(device) for e in sampled], dim = 0)

            features, _ = feature_ext(meta_obs)
            logits = policy_module(features.reshape(-1, features.size(-1)))
            dist = torch.distributions.Categorical(logits = logits)

            log_probs = dist.log_prob(meta_act)
            entropy = dist.entropy().mean()

            # Use intrinsic reward
            norm_rnd_reward = meta_rew  # already computed in generate_meta_episode

            # Compute loss (REINFORCE-style for actor + RND)
            actor_loss = -(log_probs * norm_rnd_reward).mean()
            total_loss = actor_loss + config["curiosity_args"]["rnd_lmbda"] * np.mean(total_rnd_losses)

            # Backpropagation
            optimiser.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(combined_net_params, config["policy_args"]["max_grad_norm"])
            optimiser.step()
            scheduler.step()

            logs["actor_loss"].append(actor_loss.item())
            logs["entropy"].append(entropy.item())
            logs["rnd_loss"].append(np.mean(total_rnd_losses))
            logs["extrinsic_reward"].append(np.mean(total_extr_rews))
            logs["norm_rnd_reward"].append(np.mean(total_norm_rnds))
            logs["raw_rnd_reward"].append(np.mean(total_raw_rnds))
            logs["lr"].append(optimiser.param_groups[0]["lr"])

            tb_logger.add_scalar("Train/actor_loss", logs["actor_loss"][-1], global_step = global_step)
            tb_logger.add_scalar("Train/entropy", logs["entropy"][-1], global_step = global_step)
            tb_logger.add_scalar("Train/rnd_loss", logs["rnd_loss"][-1], global_step = global_step)
            tb_logger.add_scalar("Train/extrinsic_reward", logs["extrinsic_reward"][-1], global_step = global_step)
            tb_logger.add_scalar("Train/norm_rnd_reward", logs["norm_rnd_reward"][-1], global_step = global_step)
            tb_logger.add_scalar("Train/raw_rnd_reward", logs["raw_rnd_reward"][-1], global_step = global_step)
            tb_logger.add_scalar("Train/lr", logs["lr"][-1], global_step = global_step)

            # Remove unused data to free up memory
            del meta_obs, meta_rew, meta_act, features, logits, dist, log_probs, norm_rnd_reward
            torch.cuda.empty_cache()

            if device == "cuda":
                torch.cuda.synchronize()

        # Cleanup episode memory
        del meta_episodes
        torch.cuda.empty_cache()

        tqdm.write(
            f"[Iter {global_step}] "
            f"Actor loss: {logs['actor_loss'][-1]:.4f} | "
            f"RND loss: {logs['rnd_loss'][-1]:.4f} | "
            f"Extr. reward: {logs['extrinsic_reward'][-1]:.4f} | "
            f"Intr. reward: {logs['norm_rnd_reward'][-1]:.4f} | "
            f"Entropy: {logs['entropy'][-1]:.4f}"
        )

        global_step += 1
        
        torch.cuda.empty_cache()
        gc.collect()

        pbar.update(1)

    tb_logger.close()

    torch.save(policy_module.state_dict(), f"{save_path}/actor_head.pth")
    torch.save(critic_module.state_dict(), f"{save_path}/critic_head.pth")
    torch.save(rnd_module.state_dict(), f"{save_path}/rnd_module.pth")
    torch.save(feature_ext.state_dict(), f"{save_path}/feature_extractor.pth")