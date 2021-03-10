#!/usr/bin/env python

import numpy as np
import os

from diy_gym import DIYGym

# from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env

if __name__ == '__main__':

	config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'touch_r2d2.yaml')
	env = DIYGym(config_file)
	env = make_vec_env(lambda: env, n_envs=1)

# #--- TRAINING ---
	# Instantiate the agent
	model = PPO2('MlpPolicy', env)
	# Checks the validity of the environment
	model.set_env(env)
	# # Train the agent
	model.learn(total_timesteps=500000) 
	# Save the agent
	model.save("mymodel_touch")

# --- enjoy the TESTING ---
	# model = PPO2.load("mymodel_touch")

	# observation = env.reset()
	# # while True:
	# for i in range(1000):
	# 	action, _states = model.predict(observation)
	# 	observation, reward, terminal, info = env.step(action)

	# 	if terminal:
	# 		observation = env.reset()
