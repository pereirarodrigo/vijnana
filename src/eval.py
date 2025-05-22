import yaml
import torch
import minigrid
import gymnasium
from utils.train_utils import preprocess_obs
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


env = gymnasium.make("MiniGrid-Empty-5x5-v0", render_mode = "human")
observation, info = env.reset()

policy_module, _, shared_module = build_actor_critic(env, config, device)
policy_module.load_state_dict(torch.load("saved_models/MiniGrid-Empty-5x5-v0_pg_rl2/actor_head.pth"))
policy_module.eval()

# Inference loop
obs, info = env.reset()
h_state = None

for _ in range(200):
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
