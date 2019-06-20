import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym

from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
from itertools import count

# define tensorboard summary
writer = SummaryWriter()

# create environment
env = gym.make("CartPole-v0")

# create actor neural network
actor = nn.Sequential(
		nn.Linear(4, 128),
		nn.ReLU(),
		nn.Linear(128, env.action_space.n)
	).cuda()


# create critic neural network
critic = nn.Sequential(
		nn.Linear(4, 128),
		nn.ReLU(),
		nn.Linear(128, 1)
	).cuda()

# define optimizers
actor_optimizer = torch.optim.RMSprop(actor.parameters(), lr=7e-4)
critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=7e-4)

# loss critierion
critic_criterion = nn.MSELoss()


def sample_trajectories(no_trajectories=128):
	trajectories = []
	returns = []

	for i in range(no_trajectories):
		state = env.reset()

		# define some buffers
		states = []; next_states = []
		actions = []; rewards = []
		masks = []; dists = []

		rreturn = 0

		for t in count():
			state = torch.tensor(state).unsqueeze(0).float().cuda()

			dist = None
			with torch.no_grad():
				dist = F.softmax(actor(state), dim=1).squeeze(0)

			# sample action
			categorical = Categorical(dist)
			action = categorical.sample().item()

			# interact with env
			next_state, reward, done, _ = env.step(action)
			rreturn += reward

			# add sample to buffers
			states.append(state)
			actions.append(torch.tensor([action]))
			rewards.append(torch.tensor([reward]))
			next_states.append(torch.tensor(next_state).unsqueeze(0).float())
			masks.append(torch.tensor([done]).float())
			dists.append(dist.unsqueeze(0))

			# update state
			state = next_state

			if done:
				trajectories.append((states, actions, rewards, next_states, masks, dists))
				returns.append(rreturn)
				break

	return trajectories, returns

def optimize_critic(trajectories, gamma=0.99):
	loss = 0

	for states, actions, rewards, next_states, masks, _ in trajectories:
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


def optimize_actor(trajectories, no_epochs=4, gamma=0.99, eps=0.2):
	for epoch in range(no_epochs):
		loss = 0

		for states, actions, rewards, next_states, masks, dists in trajectories:
			states = torch.cat(states, dim=0)
			actions = torch.cat(actions).reshape(-1, 1).cuda()
			rewards = torch.cat(rewards).reshape(-1, 1).cuda()
			next_states = torch.cat(next_states, dim=0).cuda()
			masks = torch.cat(masks).reshape(-1, 1).cuda()
			dists = torch.cat(dists, dim=0)

			# compute log probabilities
			pi = F.softmax(actor(states), dim=1).gather(1, actions)
			log_pi = torch.log(pi)
			old_pi = dists.gather(1, actions)

			# compute advantage
			adv = rewards + gamma * (1. - masks) * critic(next_states) - critic(states)

			# compute loss
			ratio1 = pi / old_pi
			surr1 = ratio1 * adv
		
			ratio2 = torch.clamp(ratio1, 1.0 - eps, 1.0 + eps)
			surr2 = ratio2 * adv

			loss += torch.sum(torch.min(surr1, surr2))

		loss = -loss / len(trajectories)

		actor_optimizer.zero_grad()
		loss.backward()
		for param in actor.parameters():
			param.grad.data.clamp_(-1, 1)
		actor_optimizer.step()



def ppo_clipped(no_updates):
	for update in range(1, no_updates):
		# sample trajectories
		trajectories, returns = sample_trajectories(256)

		# compute gradient and optimize critic
		optimize_critic(trajectories)

		# optimize actor
		optimize_actor(trajectories)

		# tensorboardX logger
		for name, param in critic.named_parameters():
			writer.add_histogram("critic/" + name, param.clone().cpu().data.numpy(), update)

		for name, param in actor.named_parameters():
			writer.add_histogram("actor/" + name, param.clone().cpu().data.numpy(), update)

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



if __name__ == "__main__":
	ppo_clipped(1000000)