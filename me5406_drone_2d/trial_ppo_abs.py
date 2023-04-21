import gymnasium as gym
from envs.drone_2d_abs_env import Drone2dAbsEnv
from stable_baselines3 import PPO
import os
import time


env = Drone2dAbsEnv(render_mode="human") # not 
observation, info = env.reset()


#model_path = "models/PPO_abs/10000000_timesteps.zip"
#model_path = "models/PPO_abs_revised/1000000_timesteps.zip"
model_path = "models/PPO_abs_revised/2000000_timesteps.zip" 
model = PPO.load(model_path, env=env)

# Show trained model
episodes = 20

for ep in range(episodes):
    print(f"Episode {ep+1}")
    obs = env.reset()
    terminated = False
    truncated = False

    while not terminated: #and not truncated:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
    # for step in range(150):
    #     action, _ = model.predict(observation)
    #     observation, reward, done, truncated, info = env.step(action)
        

    
env.close()









