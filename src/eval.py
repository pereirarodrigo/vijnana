import yaml
import torch
import minigrid
import gymnasium
from utils import preprocess_obs
from modelling.policy import build_policy
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


# Load the config (YAML) file
with open("src/best_params_so_far.yml", 'r') as file:
    base_config = yaml.safe_load(file)

# Define the device
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Set the number of evaluation episodes
num_eval_episodes = 200

# Set the environment of interest
env_name = "MiniGrid-Unlock-v0"
config = base_config[env_name]

# Setting up the env and recording options
env = gymnasium.make(env_name, render_mode = "rgb_array")
env = RecordVideo(
    env, 
    video_folder = "recordings/minigrid-unlock-agent", 
    name_prefix = "eval",
    episode_trigger = lambda x: True
)
env = RecordEpisodeStatistics(env)
observation, info = env.reset()

policy_module, shared_module = build_policy(env, config, device)
policy_module.load_state_dict(torch.load("saved_models/MiniGrid-Unlock-v0_pg_rl2/actor_head.pth"))
policy_module.eval()

# Inference loop
obs, info = env.reset()
h_state = None

for _ in range(num_eval_episodes):
    env.render()
    obs_tensor = preprocess_obs(observation, device)

    with torch.no_grad():
        features, h_state = shared_module(obs_tensor.unsqueeze(1), h_state)
        logits = policy_module(features[:, -1])
        dist = torch.distributions.Categorical(logits = logits)
        action = dist.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        h_state = None
        observation, info = env.reset()

env.close()
