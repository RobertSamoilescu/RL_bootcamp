import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action='store_true', help="Show plot while training")

args = parser.parse_args()

def sample_action(q_table, state, n_actions, eps=0.5):
    q = q_table.get((state, 0), 0)
    best_action = 0

    for action in range(1, n_actions):
        if q_table.get((state, action), 0) > q:
            q = q_table.get((state, action), 0)
            best_action = action

    if np.random.rand() >= eps:
        return best_action

    return np.random.choice(np.arange(n_actions))


def sarsa(no_episodes=1000, alpha=.4, gamma=.99, verbose=True):
    # table of state-action pairs
    q_table = {}

    # initialize eps
    max_eps = 1.0
    min_eps = 0.01
    eps = max_eps

    # epsilon decay rate
    decay_rate = 0.01

    # make env
    env = gym.make('Taxi-v2')
    n_states, n_actions = env.observation_space.n, env.action_space.n

    # store all scores
    episodes_rewards = []

    for i in tqdm(range(no_episodes)):
        state = env.reset()
        
        total_reward = 0

        eps = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * i)

        while True:
            # sample action
            action = sample_action(q_table, state, n_actions, eps=eps)
            
            # observe next state and reward
            next_state, r, done, _ = env.step(action)

            # update total reward
            total_reward += r

            # get best action for next_state
            next_action = sample_action(q_table, state, n_actions, eps=.0)

            # update q_table
            if not done:
                q_table[(state, action)] = (1 - alpha) * q_table.get((state, action), 0) + \
                    alpha * (r + gamma * q_table.get((next_state, next_action), 0))
            else:
                q_table[(state, action)] = (1 - alpha) * q_table.get((state, action), 0) + \
                    alpha * r

            # update next_state
            state = next_state

            if done:
                episodes_rewards.append(total_reward)
               
                if verbose:
                    plt.plot(episodes_rewards[-100:], 'b')
                    plt.title("Train")
                    plt.draw()
                    plt.pause(0.1)
                    plt.clf()
                break

    plt.plot(episodes_rewards)
    plt.show()

    return q_table


def test(q_table, no_episodes):
    # make env
    env = gym.make('Taxi-v2')
    n_states, n_actions = env.observation_space.n, env.action_space.n

    # store all scores
    episodes_rewards = []

    for i in tqdm(range(no_episodes)):
        state = env.reset()
        total_reward = 0

        while True:
            # sample action
            action = sample_action(q_table, state, n_actions, eps=.0)
            
            # observe next state and reward
            next_state, r, done, _ = env.step(action)
            env.render()

            # update total reward
            total_reward += r

            # update next_state
            state = next_state

            if done:
                print("Total score: %.2f" % (total_reward, ))   
                episodes_rewards.append(total_reward)
                break

            
    plt.plot(episodes_rewards)
    plt.title("Test")
    plt.show()


def main():
    # Q-Learning
    q_table = sarsa(no_episodes=10000, verbose=args.verbose)

    # test policy
    test(q_table, 100)


if __name__ == "__main__":
    main()