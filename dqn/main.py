from buffer import *
from model import *
from utils import *

from copy import deepcopy
from itertools import count

import gym
import torch.optim as optim
import matplotlib.pyplot as plt

# define environment
env = gym.make('CartPole-v0').unwrapped

# define replay buffer
buff = ReplayBuffer()

# define networks
nn = DQN(40, 90, env.action_space.n).cuda()
targe_nn = DQN(40, 90, env.action_space.n).cuda()
targe_nn.load_state_dict(nn.state_dict())
targe_nn.eval()

# define optimizer
optimizer = optim.RMSprop(nn.parameters())

def sample_action(state, nn, eps=.1):
    state = torch.from_numpy(state).unsqueeze(0).cuda()

    # pass state through nn
    with torch.no_grad():
        q = nn(state).squeeze().cpu().numpy()

    # choose best action
    if np.random.rand() >= eps:
        return np.argmax(q)

    # choose random action
    return np.random.choice(np.arange(env.action_space.n))


def update_model(nn, targe_nn, batch_size=128, gamma=0.999):
    if batch_size > len(buff):
        return

    # sample experience from replay buffer
    samples = Transition(*zip(*buff.sample(batch_size)))

    # construct batches
    state_batch = torch.cat(samples.state).cuda()
    action_batch = torch.cat(samples.action).cuda()
    reward_batch = torch.cat(samples.reward).cuda()
    next_state_batch = torch.cat(samples.next_state).cuda()
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
    mean_scores = []
    step = 0

    for i in range(1, no_episodes + 1):
        # update eps
        eps = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * i)

        # print log
        if i % 10 == 0:
            print("Episode: %d, Eps: %.4f" % (i, eps))

        # reset env
        env.reset()

        # initialize previous & current screen
        prev_screen = get_screen(env)
        screen = get_screen(env)

        # define total_reward
        total_reward = 0

        for t in count():
            # compute state as difference between screen and prev_screen
            state = screen - prev_screen

            # sample action epsilon greedy
            action = sample_action(state, nn, eps=eps)

            # execute action in env
            _, reward, done, _ = env.step(action)
            total_reward += reward

            # get new observation & compute new state
            prev_screen = screen
            screen = get_screen(env)

            new_state = screen - prev_screen

            # store state
            buff.append(([state], [[action]], [[reward]], [new_state], [[int(done)]]))

            # update nn
            update_model(nn, targe_nn)

            if done:
                scores.append(total_reward)
                mean_scores.append(np.mean(scores[-100:]))
                
                # plot scores
                plt.plot(scores[-200:], 'b')
                plt.plot(mean_scores[-200:], 'r')
                plt.draw()
                plt.pause(0.1)
                plt.clf()

                break

        if i % target_update == 0:
            targe_nn.load_state_dict(nn.state_dict())
            print("Model updated")

    env.close()

    plt.plot(scores)
    plt.show()


if __name__ == "__main__":
    dqn(no_episodes=100000, target_update=100)