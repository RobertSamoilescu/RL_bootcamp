import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


import csv
import cv2
import numpy as np
import gym

from itertools import count
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter


# define tensorboard summary
writer = SummaryWriter()

# create environment
env = gym.make("CartPole-v0")

# create model
policy_nn = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n)
    ).cuda()

optimizer = torch.optim.RMSprop(policy_nn.parameters(), lr=7e-4)

def sample_trajectories(no_trajectories=256, gamma=0.99):
    trajectories = []
    returns = []

    for i in range(no_trajectories):
        state = env.reset()

        states = []; actions = []; rewards = []

        rreturn = 0

        for t in count():
            state = torch.tensor(state).unsqueeze(0).float().cuda()

            with torch.no_grad():
                probs = F.softmax(policy_nn(state), dim=1).squeeze(0)

            # sample action
            categorical = Categorical(probs)
            action = categorical.sample().item()
           
            # interact with env
            new_state, reward, done, info = env.step(action)
            rreturn += reward

            # add sample to trajectory
            states.append(state)
            actions.append(torch.tensor([action]))
            rewards.append(torch.tensor([reward]))

            # update state
            state = new_state

            if done:
                running_reward = 0

                for j in reversed(range(0, len(rewards))): 
                    running_reward = running_reward * gamma + rewards[j]
                    rewards[j] = running_reward

                trajectories.append((states, actions, rewards))
                returns.append(rreturn)

                break
    
    return trajectories, returns


def optimize(trajectories):
    loss = 0

    for s, a, dr in trajectories:
        s = torch.cat(s, dim=0)
        a = torch.cat(a).reshape(-1, 1).cuda()        
        dr = torch.cat(dr).reshape(-1, 1).cuda()

        log_pi = torch.log(F.softmax(policy_nn(s), dim=1).gather(1, a))
        loss += torch.sum(log_pi * dr)

    loss = -loss / len(trajectories)
    

    optimizer.zero_grad()
    loss.backward()
    for param in policy_nn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def reinforce(no_updates):
    for update in range(1, no_updates + 1):
    	# sample trajectories
        trajectories, returns = sample_trajectories()
       	
       	# compute gradients and optimize model
        optimize(trajectories)

        # tensorboardx logger
        for name, param in policy_nn.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), update)
      
        writer.add_scalar("mean_retrun", np.mean(returns), update)
        writer.add_scalar("min_return", np.min(returns), update)
        writer.add_scalar("max_return", np.max(returns), update)

        if update % 10 == 0:
        	torch.save(policy_nn.state_dict(), "model")
        	print("Model saved")

    env.close()
    writer.close()


if __name__ == "__main__":
    reinforce(no_updates=10000000)
