from tqdm import tqdm
from collections import defaultdict

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR   
from modelling.reward_norm import IntrinsicRewardNormaliser
from utils.train_utils import compute_intrinsic_reward, preprocess_obs


def train_policy(
    env: gym.Env,
    policy_module: nn.Module,
    critic_module: nn.Module,
    shared_module: nn.Module,
    rnd_module: nn.Module,
    intrinsic_norm: IntrinsicRewardNormaliser,
    optimiser: Optimizer,
    scheduler: CosineAnnealingLR,
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
    num_iters = config["exp_args"]["num_steps"]
    rollout_len = config["exp_args"]["rollout_steps"]

    for i in range(num_iters):
        obs_seq, act_seq, rew_seq, done_seq = [], [], [], []
        h_state = None
        obs, _ = env.reset()

        for _ in range(rollout_len):
            # Preprocess obs due to Minigrid's use of dictionaries
            obs_tensor = preprocess_obs(obs, device)

            # Perform a pass in the policy
            with torch.no_grad():
                features, h_state = shared_module(obs_tensor.unsqueeze(1), h_state)
                logits = policy_module(features[:, -1])
                dist = torch.distributions.Categorical(logits = logits)
                action = dist.sample()

            next_obs, reward, done, _, _ = env.step(action)
            obs_seq.append(obs_tensor.squeeze(0))
            act_seq.append(action)
            rew_seq.append(torch.tensor([reward], dtype = torch.float32))
            done_seq.append(torch.tensor([done], dtype = torch.float32))

            obs = next_obs

            if done:
                h_state = None
                obs, _ = env.reset()

        # Convert observations and rewards to tensors
        obs_tensor_seq = torch.stack(obs_seq).unsqueeze(0).to(device)
        rew_tensor_seq = torch.stack(rew_seq).to(device)

        # Extract features
        with torch.no_grad():    
            feats, _ = shared_module(obs_tensor_seq)
            
        # Compute intrinsic rewards
        targets, preds = rnd_module(feats.squeeze(0))
        raw_rnd_reward = compute_intrinsic_reward(preds, targets).detach()

        intrinsic_norm.update(raw_rnd_reward)   

        norm_rnd_reward = intrinsic_norm.normalise(raw_rnd_reward)

        # Total loss (curiosity + action loss)
        actor_logits = policy_module(feats[:, -1])
        dist = torch.distributions.Categorical(logits = actor_logits)
        actions = torch.stack(act_seq).to(device)
        log_probs = dist.log_prob(actions)

        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * norm_rnd_reward).mean()
        rnd_loss = nn.MSELoss()(preds, targets)
        total_loss = actor_loss + config["curiosity_args"]["rnd_lmbda"] * rnd_loss

        # Backpropagation
        optimiser.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(combined_net_params, config["policy_args"]["max_grad_norm"])
        optimiser.step()
        scheduler.step()

        # Logging
        logs["rnd_loss"].append(rnd_loss.item())
        logs["actor_loss"].append(actor_loss.item())
        logs["entropy"].append(entropy.item())
        logs["reward"].append(rew_tensor_seq.mean().item())
        logs["norm_rnd_reward"].append(norm_rnd_reward.mean().item())
        logs["raw_rnd_reward"].append(raw_rnd_reward.mean().item())
        logs["lr"].append(optimiser.param_groups[0]["lr"])

        tb_logger.add_scalar("Train/rnd_loss", logs["rnd_loss"][-1], global_step = i)
        tb_logger.add_scalar("Train/actor_loss", logs["actor_loss"][-1], global_step = i)
        tb_logger.add_scalar("Train/entropy", logs["entropy"][-1], global_step = i)
        tb_logger.add_scalar("Train/reward", logs["reward"][-1], global_step = i)
        tb_logger.add_scalar("Train/raw_rnd_reward", logs["raw_rnd_reward"][-1], global_step = i)
        tb_logger.add_scalar("Train/norm_rnd_reward", logs["norm_rnd_reward"][-1], global_step = i)
        tb_logger.add_scalar("Train/lr", logs["lr"][-1], global_step = i)

        tqdm.write(
            f"[Iter {i}]\n \
            ----------------\n \
            Actor loss: {logs['actor_loss'][-1]:.4f}\n \
            RND loss: {logs['rnd_loss'][-1]:.4f}\n \
            Reward: {logs['reward'][-1]:.4f}\n \
            Intrinsic reward: {logs['norm_rnd_reward'][-1]:.4f}\n \
            Entropy: {logs['entropy'][-1]:.4f}\n"
        )

        pbar.update(1)

    tb_logger.close()

    torch.save(policy_module.state_dict(), f"{save_path}/actor_head.pth")
    torch.save(critic_module.state_dict(), f"{save_path}/critic_head.pth")
    torch.save(rnd_module.state_dict(), f"{save_path}/rnd_module.pth")
    torch.save(shared_module.state_dict(), f"{save_path}/feature_extractor.pth")