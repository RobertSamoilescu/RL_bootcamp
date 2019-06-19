from buffer import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from itertools import count
from tensorboardX import SummaryWriter

import gym
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import csv

# define tensorboard summary
writer = SummaryWriter()

# define environment
env = gym.make('CartPole-v0')

# define replay buffer
buff = ReplayBuffer()

# define model class
class DQN(nn.Module):
    def __init__(self, no_inputs, no_outputs):
        super(DQN, self).__init__()

        self.l1 = nn.Linear(no_inputs, 128)
        self.l2 = nn.Linear(128, no_outputs)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

# define neural networks
policy_nn = DQN(4, env.action_space.n).cuda()
targe_nn = DQN(4, env.action_space.n).cuda()

targe_nn.load_state_dict(policy_nn.state_dict())
targe_nn.eval()

# define optimizer
optimizer = optim.RMSprop(policy_nn.parameters(), lr=7e-4)

def sample_action(state, policy_nn, eps=.1):
    state = torch.from_numpy(state).unsqueeze(0).cuda().float()

    # pass state through nn
    with torch.no_grad():
        q = policy_nn(state).squeeze().cpu().numpy()

    # choose best action
    if np.random.rand() >= eps:
        return np.argmax(q)

    # choose random action
    return np.random.choice(np.arange(env.action_space.n))


def update_model(policy_nn, targe_nn, batch_size=128, gamma=0.999):
    if batch_size > len(buff):
        return

    # sample experience from replay buffer
    samples = Transition(*zip(*buff.sample(batch_size)))

    # construct batches
    state_batch = torch.cat(samples.state).float().cuda()
    action_batch = torch.cat(samples.action).cuda()
    reward_batch = torch.cat(samples.reward).cuda()
    next_state_batch = torch.cat(samples.next_state).float().cuda()
    done_batch =  torch.cat(samples.done).float().cuda()

    # compute target
    with torch.no_grad():
        y_target_batch = reward_batch + (1. - done_batch) * gamma * targe_nn(next_state_batch).max(1)[0].reshape(-1, 1)

    # compute q values for batch actions
    y_batch = policy_nn(state_batch).gather(1, action_batch)

    # compute loss
    loss = F.smooth_l1_loss(y_batch, y_target_batch)

    # optimize model
    optimizer.zero_grad()
    
    loss.backward()

    for param in policy_nn.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()



def dqn(no_episodes, target_update, max_eps=.9, min_eps=0.05, decay_rate=.005):
    scores = []

    for i in range(1, no_episodes + 1):
        # update eps
        eps = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * i)

        # reset env
        state = env.reset()
        
        # define rreturn
        rreturn = 0

        for t in count():
            env.render()

            # sample action epsilon greedy
            action = sample_action(state, policy_nn, eps=eps)

            # execute action in env
            new_state, reward, done, _ = env.step(action)

            # update total reward
            rreturn += reward

            # store state
            buff.append(([state], [[action]], [[reward]], [new_state], [[int(done)]]))

            # update state
            state = new_state

            # update policy_nn
            update_model(policy_nn, targe_nn)

            if done:
                scores.append(rreturn)
                scores = scores[-100:]

                # tensorboard logger
                writer.add_scalar('mean_return', np.mean(scores[-100:]), i)
                writer.add_scalar('min_return', np.min(scores[-100:]), i)
                writer.add_scalar('max_return', np.max(scores[-100:]), i)

                break

        # save model        
        if i % 10 == 0:
            torch.save(policy_nn.state_dict(), "model")

        if i % target_update == 0:
            # target update
            targe_nn.load_state_dict(policy_nn.state_dict())

            # tensorboard logger
            for name, param in targe_nn.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), i // target_update)
            
            print("Model updated")

    env.close()
    logger.close()
    writer.close()

    # plt.plot(scores)
    # plt.show()


if __name__ == "__main__":
    dqn(no_episodes=10000000, target_update=100)