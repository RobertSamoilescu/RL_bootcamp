from env import *
import matplotlib.pyplot as plt


def value_iteration(no_iter=100, gamma=0.9):
    # initialize variable
    V = np.zeros_like(rewards).reshape(-1, 1)
    R = rewards.reshape(-1, 1)

    for i in range(no_iter):
        # compute next value function for every action
        V_next = []
        
        for a in actions:
            V_next.append(np.dot(T[a], V))

        # stack them all and take the maximum
        V_next = np.hstack(V_next)
        V_next = np.max(V_next, axis=1).reshape(-1, 1)

        # make update
        V = R + gamma * V_next

    return V


def show_map(V):
    grid = np.zeros((3, 4))
    it = 0

    # fill grid with value function
    for i in range(2, -1, -1):
        for j in range(4):
            if i == 1 and j == 1:
                continue

            grid[i, j] = V[it]
            it += 1

    # display map
    fig, ax = plt.subplots()
    ax.matshow(grid, cmap=plt.cm.Blues)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            c = grid[i, j]
            ax.text(j, i, "{:.2f}".format(c), va='center', ha='center')
    
    plt.show()

    
def main():
    V = value_iteration()
    show_map(V)

if __name__ == "__main__":
    main()