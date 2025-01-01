import gymnasium as gym
from envs.drone_2d_abs_env import Drone2dAbsEnv
from stable_baselines3 import PPO
import os
import time


env = Drone2dAbsEnv(render_mode="human") 
observation, info = env.reset()

model_path = "models/PPO_abs_revised/11000000_timesteps.zip" 
model = PPO.load(model_path, env=env)

# Show trained model
episodes = 20

for ep in range(episodes):
    print(f"Episode {ep+1}")
    obs = env.fair_reset()
    terminated = False
    truncated = False

    while not terminated: #and not truncated:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
    
    if info["state"] == "goal_reached":
        print("Drone stabilized")
        # Keep going a bit longer
        for i in range(60): # 3s
            action, _ = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if i == 20:
                print("New disturbance in 2")
            if i == 40:
                print("New disturbance in 1")
    else:
        print("Drone left viewport")
env.close()








