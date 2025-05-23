import tqdm

import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from utils.train_utils import preprocess_obs, compute_intrinsic_reward
from modelling.reward_norm import IntrinsicRewardNormaliser

from typing import List, Tuple


@torch.no_grad()
def generate_meta_episode(
    env: gym.Env, 
    policy_module: nn.Module, 
    critic_module: nn.Module | None,
    feature_ext: nn.Module,
    rnd_module: nn.Module, 
    intrinsic_norm: IntrinsicRewardNormaliser,
    episode_len: int,
    rollout_len: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List, List, List]:
    """
    Generate a meta-episode consisting of episode_len rollouts of rollout_len each.
    """
    ep_obs, ep_actions, ep_rewards, ep_ext_rewards = [], [], [], []
    rnd_losses, raw_rnds, norm_rnds = [], [], []
    
    for _ in range(episode_len):
        obs_seq, act_seq, rew_seq = [], [], []
        ext_rew_episode = []
        h_state = None
        obs, _ = env.reset()

        for _ in range(rollout_len):
            obs_tensor = preprocess_obs(obs, device)
            features, h_state = feature_ext(obs_tensor.unsqueeze(1), h_state)
            logits = policy_module(features[:, -1])
            dist = torch.distributions.Categorical(logits = logits)
            action = dist.sample()

            next_obs, rewards, done, _, _ = env.step(action)
            
            # Append rollout data
            obs_seq.append(obs_tensor.squeeze(0))
            ext_rew_episode.append(torch.tensor(rewards, dtype=torch.float32).unsqueeze(0))

            # Ensure action is scalar tensor
            if isinstance(action, torch.Tensor):
                action = action.squeeze()

                if action.dim() != 0:
                    action = action[0]  # fallback if batch dimension remains
                    
            else:
                action = torch.tensor(action)

            act_seq.append(action)

            # Convert observations to tensors
            obs_tensor_seq = torch.stack(obs_seq).unsqueeze(0).to(device)

            # Intrinsic reward
            with torch.no_grad():
                feats, _ = feature_ext(obs_tensor_seq)
            
            targets, preds = rnd_module(feats.squeeze(0))
            intrinsic_r = compute_intrinsic_reward(preds, targets).detach()
            raw_rnd = compute_intrinsic_reward(preds, targets).detach()
            rnd_loss = nn.MSELoss()(preds, targets)

            intrinsic_norm.update(intrinsic_r)

            norm_rnd = intrinsic_norm.normalise(raw_rnd)

            # Append intrinsic rewards
            rew_seq.append(intrinsic_r[-1].unsqueeze(0))

            # Append RND-specific data
            raw_rnds.append(raw_rnd.mean().item())
            norm_rnds.append(norm_rnd.mean().item())
            rnd_losses.append(rnd_loss.item())

            obs = next_obs

            #  Free memory for each step
            del obs_tensor, features, logits, dist, action, obs_tensor_seq, feats, targets, preds
            torch.cuda.empty_cache()
            
            if device == "cuda":
                torch.cuda.synchronize()

            if done:
                break

        # Verify length for padding
        actual_len = len(obs_seq)
        pad_len = rollout_len - actual_len

        # Stack tensors
        obs_tensor = torch.stack(obs_seq)                      # [T, C, H, W]
        rew_tensor = torch.stack(rew_seq)                      # [T, 1]
        act_tensor = torch.stack(act_seq)                      # [T]
        ext_rew_tensor = torch.stack(ext_rew_episode)          # [T, 1]

        # Pad tensors to ensure data uniformity
        if pad_len > 0:
            obs_tensor = F.pad(obs_tensor, (0, 0, 0, 0, 0, 0, 0, pad_len))   # pad time dim
            rew_tensor = F.pad(rew_tensor, (0, 0, 0, pad_len))
            act_tensor = F.pad(act_tensor, (0, pad_len))
            ext_rew_tensor = F.pad(ext_rew_tensor, (0, 0, 0, pad_len))

        ep_obs.append(obs_tensor)      # [rollout_len, C, H, W]
        ep_rewards.append(rew_tensor)  # [rollout_len, 1]
        ep_actions.append(act_tensor)  # [rollout_len]
        ep_ext_rewards.append(ext_rew_tensor)

        # Clear per-episode lists
        del obs_seq, rew_seq, act_seq, obs_tensor, rew_tensor, act_tensor
        torch.cuda.empty_cache()
        
        if device == "cuda":
            torch.cuda.synchronize()

    # Final stacking
    ep_obs = torch.stack(ep_obs).to(device)                 # [episode_len, rollout_len, C, H, W]
    ep_rewards = torch.stack(ep_rewards).to(device)         # [episode_len, rollout_len, 1]
    ep_ext_rewards = torch.stack(ep_ext_rewards).to(device) # [episode_len, rollout_len, 1]
    ep_ext_rewards = ep_ext_rewards.view(-1)                # [episode_len * rollout_len]

    # Flatten actions to 1D to match Categorical requirements
    ep_actions = torch.stack(ep_actions)               # [episode_len, rollout_len]
    ep_actions = ep_actions.view(-1).to(device)        # [episode_len * rollout_len]

    # Last cleanup
    torch.cuda.empty_cache()

    if device == "cuda":
        torch.cuda.synchronize()

    return ep_obs, ep_rewards, ep_actions, ep_ext_rewards, rnd_losses, raw_rnds, norm_rnds