import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import numpy as np
import gym

from itertools import count
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter

# define tensorboard summary
writer = SummaryWriter()

# create environment
env = gym.make("CartPole-v0")

# create actor model
actor = nn.Sequential(
		nn.Linear(4, 128),
		nn.ReLU(),
		nn.Linear(128, env.action_space.n)
	).cuda()

# create critic model
critic = nn.Sequential(
		nn.Linear(4, 128),
		nn.ReLU(),
		nn.Linear(128, 1)
	).cuda()

# define optimizers
actor_optimizer = torch.optim.RMSprop(actor.parameters(), lr=7e-4)
critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=7e-4)

# loss criterion
critic_criterion = nn.MSELoss()


def sample_trajectories(no_trajectories=256):
	trajectories = []
	returns = []

	for i in range(no_trajectories):
		state = env.reset()

		states = []; next_states = []
		actions = []; rewards = []
		masks = []

		rreturn = 0

		for t in count():
			state = torch.tensor(state).unsqueeze(0).float().cuda()

			with torch.no_grad():
				probs = F.softmax(actor(state), dim=1).squeeze(0)

			# sample action
			categorical = Categorical(probs)
			action = categorical.sample().item()

			# interact with env
			next_state, reward, done, info = env.step(action)
			rreturn += reward

			# add sample to trajectory
			states.append(state)
			actions.append(torch.tensor([action]))
			rewards.append(torch.tensor([reward]))
			next_states.append(torch.tensor(next_state).unsqueeze(0).float())
			masks.append(torch.tensor([done]).float())

			# update state
			state = next_state

			if done:
				trajectories.append((states, actions, rewards, next_states, masks))
				returns.append(rreturn)
				break

	return trajectories, returns

def optimize_critic(trajectories, gamma=0.99):
	loss = 0

	for states, actions, rewards, next_states, masks in trajectories:
		states = torch.cat(states, dim=0)
		actions = torch.cat(actions).reshape(-1, 1).cuda()
		rewards = torch.cat(rewards).reshape(-1, 1).cuda()
		next_states = torch.cat(next_states, dim=0).cuda()
		masks = torch.cat(masks).reshape(-1, 1).cuda()

		y = critic(states)
		y_target = rewards + gamma * (1. - masks) * critic(next_states)

		loss += critic_criterion(y, y_target)

	loss = loss / len(trajectories)

	# optimize critic
	critic_optimizer.zero_grad()
	loss.backward()
	for param in critic.parameters():
		param.grad.data.clamp(-1, 1)
	critic_optimizer.step()


def optimize_actor(trajectories, gamma=0.99):
	loss = 0

	for states, actions, rewards, next_states, masks in trajectories:
		states = torch.cat(states, dim=0)
		actions = torch.cat(actions).reshape(-1, 1).cuda()
		rewards = torch.cat(rewards).reshape(-1, 1).cuda()
		next_states = torch.cat(next_states, dim=0).cuda()
		masks = torch.cat(masks).reshape(-1, 1).cuda()

		# compute log probabilities
		log_pi = torch.log(F.softmax(actor(states), dim=1).gather(1, actions))

		# compute advantage
		adv = rewards + gamma * (1. - masks) * critic(next_states) - critic(states)

		# compute loss
		loss += torch.sum(log_pi * adv)

	loss = -loss / len(trajectories)

	actor_optimizer.zero_grad()
	loss.backward()
	for param in actor.parameters():
		param.grad.data.clamp_(-1, 1)
	actor_optimizer.step()


def actor_critic(no_updates):
	for update in range(1, no_updates + 1):
		# sample trajectories
		trajectories, returns = sample_trajectories()

		# compute gradient and optimize critic
		optimize_critic(trajectories)

		# compute gradient and optimize actor
		optimize_actor(trajectories)

		# tensorboardX logger
		for name, param in critic.named_parameters():
			writer.add_histogram(name, param.clone().cpu().data.numpy(), update)

		for name, param in actor.named_parameters():
			writer.add_histogram(name, param.clone().cpu().data.numpy(), update)

		min_rr, mean_rr, max_rr, std = np.min(returns), np.mean(returns), \
			np.max(returns), np.std(returns)


		# logs
		writer.add_scalar("mean_return", mean_rr, update)
		writer.add_scalar("min_return", min_rr, update)
		writer.add_scalar("max_return", max_rr, update)

		print("Update: %d, Mean return: %.2f, Min return: %.2f, Max return: %.2f, Std: %.2f" %
			(update, mean_rr, min_rr, max_rr, std))

		if update % 10 == 0:
			torch.save(actor.state_dict(), "actor")
			torch.save(critic.state_dict(), "critic")
			print("Models saved")

	env.close()
	writer.close()

if __name__ == "__main__":
	actor_critic(no_updates=1000000)