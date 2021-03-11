#!/usr/bin/env python

import numpy as np
import os
import torch

from diy_gym import DIYGym

from stable_baselines.common.cmd_util import make_vec_env

from custom_algo.TD3 import TD3 
from custom_algo.TD3 import utils 

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	# print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("Evaluation over"+str(eval_episodes)+" episodes: "+str(avg_reward))
	print("---------------------------------------")
	return avg_reward

if __name__ == '__main__':

	# config params
	episode_reward 		= 0
	episode_timesteps 	= 0
	episode_num 		= 0
	seed 				= 0
	start_timesteps 	= 25e3
	max_timesteps 		= 1e5 	 # Max time steps to run environment
	eval_freq 			= 5e3	 # How often (time steps) we evaluate
	expl_noise 			= 0.1	 # Std of Gaussian exploration noise
	batch_size			= 256	 # Batch size for both actor and critic
	discount 			= 0.99	 # Discount factor
	tau 				= 0.005  # Target network update rate
	policy_noise 		= 0.2    # Noise added to target policy during critic update
	noise_clip 			= 0.5    # Range to clip target policy noise
	policy_freq 		= 2		 # Frequency of delayed policy updates

	# 1. env
	config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'touch_r2d2.yaml')
	env = DIYGym(config_file)
	env = make_vec_env(lambda: env, n_envs=1)

	# 2. seed
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	# 3. policy
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	print(state_dim)
	print(action_dim)
	print(max_action)
	
	kwargs = {
		"state_dim"	: state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount"	: discount,
		"tau"		: tau,
	}
	kwargs["policy_noise"] 	= policy_noise * max_action
	kwargs["noise_clip"] 	= noise_clip * max_action
	kwargs["policy_freq"] 	= policy_freq

	policy = TD3.TD3(**kwargs)


	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# 4. training

	state, done = env.reset(), False

	for t in range(int(max_timesteps)):
		
		episode_timesteps += 1

		# Select action according to policy
		action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= start_timesteps:
			policy.train(replay_buffer, batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			# print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			print("Total T: "+str(t+1)+" Episode Num: "+str(episode_num+1)+" Episode T: "+str(episode_timesteps)+ " Reward: " + str(episode_reward))
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % eval_freq == 0:
			evaluations.append(eval_policy(policy, env, seed))
			np.save("./models/"+str(file_name), evaluations)
			policy.save("./models/"+str(file_name))
