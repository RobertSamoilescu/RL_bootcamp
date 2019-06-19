from buffer import *
from model import *
from utils import *

from copy import deepcopy
from itertools import count

import gym
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import csv

# define csv for logging
file = open("logs.csv", 'w', newline='')
logger = csv.writer(file, delimiter=' ')
logger.writerow(['episode', 'min_return', 'mean_return', 'max_return', 'std'])

# define environment
env = gym.make('Breakout-v0')

# define replay buffer
buff = ReplayBuffer()

# define networks
nn = DQN(84, 64, env.action_space.n).cuda()
targe_nn = DQN(84, 64, env.action_space.n).cuda()
targe_nn.load_state_dict(nn.state_dict())
targe_nn.eval()

# define optimizer
optimizer = optim.RMSprop(nn.parameters(), lr=7e-4)

def sample_action(state, nn, eps=.1):
    state = torch.from_numpy(state).unsqueeze(0).cuda().float()

    # pass state through nn
    with torch.no_grad():
        q = nn(state).squeeze().cpu().numpy()

    # choose best action
    if np.random.rand() >= eps:
        return np.argmax(q)

    # choose random action
    return np.random.choice(np.arange(env.action_space.n))


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, None, fx=.4, fy=.4)
    frame = (frame - 128.) / 128.

    return frame


def update_model(nn, targe_nn, batch_size=128, gamma=0.999):
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
    y_batch = nn(state_batch).gather(1, action_batch)

    # compute loss
    loss = F.smooth_l1_loss(y_batch, y_target_batch)

    # optimize model
    optimizer.zero_grad()
    
    loss.backward()

    for param in nn.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()



def dqn(no_episodes, target_update, max_eps=.9, min_eps=0.05, decay_rate=.01):
    scores = []

    for i in range(1, no_episodes + 1):
        # update eps
        eps = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * i)

        # reset env
        frame = env.reset()
        frame = preprocess_frame(frame)
        
        # initialize frame buffer
        frame_buff = [frame] * 4

        # define rreturn
        rreturn = 0

        for t in count():
            # compute state as difference between screen and prev_screen
            state = np.array(frame_buff[-4:])

            # sample action epsilon greedy
            action = sample_action(state, nn, eps=eps)

            # execute action in env
            frame, reward, done, _ = env.step(action)
            frame = preprocess_frame(frame)

            # update total reward
            rreturn += reward

            # add frame to the buffer
            frame_buff.append(frame)
            
            # get new state
            new_state = np.array(frame_buff[-4:])

            # store state
            buff.append(([state], [[action]], [[reward]], [new_state], [[int(done)]]))

            # update nn
            update_model(nn, targe_nn)

            if done:
                scores.append(rreturn)
                scores = scores[-100:]

                logger.writerow([
                    i, 
                    "%.3f" % (np.min(scores[-100:]),), 
                    "%.3f" % (np.mean(scores[-100:]),),
                    "%.3f" % (np.max(scores[-100:]),),
                    "%.3f" % (np.std(scores[-100:]),)
                ])
                
                file.flush()

                # # plot scores
                # plt.plot(scores[-200:], 'b')
                # plt.plot(mean_scores[-200:], 'r')
                # plt.draw()
                # plt.pause(0.1)
                # plt.clf()

                break

        # print log
        if i % 10 == 0:
            torch.save(nn.state_dict(), "model")

        # target update
        if i % target_update == 0:
            targe_nn.load_state_dict(nn.state_dict())
            print("Model updated")

    env.close()
    logger.close()

    # plt.plot(scores)
    # plt.show()


if __name__ == "__main__":
    dqn(no_episodes=10000000, target_update=1000)