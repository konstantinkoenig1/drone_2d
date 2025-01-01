import gymnasium as gym
from envs.drone_2d_abs_env import Drone2dAbsEnv
from stable_baselines3 import PPO
import os

env = Drone2dAbsEnv() # not render_mode="human"

# For saving model and for tensorboard:
models_dir  = "models/PPO_abs_revised"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


# Train the model for 100k time steps. Then maybe longer
TIMESTEPS = 1000000 # 1 Mio * 50 Timesteps
model = PPO("MlpPolicy", env, tensorboard_log=logdir)

for i in range(1,51): 
    model.learn(total_timesteps=TIMESTEPS, 
                reset_num_timesteps=False,
                tb_log_name="revised rewards, ppo")
    model.save(f"{models_dir}/{TIMESTEPS*i}_timesteps")

