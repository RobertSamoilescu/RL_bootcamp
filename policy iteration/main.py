from env import * 
from copy import deepcopy


def policy_evaluation(pi, gamma=.9, no_iter=10):
    R = rewards.reshape(-1, 1)
    V_old = np.zeros_like(R)
    V = np.zeros_like(R)

    # evaluate current policy
    for i in range(no_iter):
        for s in states:
            V[s, 0]  = R[s, 0] + gamma * np.dot(T[pi[s, 0]][s], V_old)

        V_old = deepcopy(V)

    return V


def improve_policy(pi, V):
    # compute next value function for every action
    V_next = []
    
    for a in actions:
        V_next.append(np.dot(T[a], V))

    # stack them all and take the maximum
    V_next = np.hstack(V_next)
    new_pi = np.argmax(V_next, axis=1).reshape(-1, 1)

    return new_pi


def show_policy():
    grid = np.zeros((3, 4), dtype=np.int)
    it = 0

    # fill grid with value function
    for i in range(2, -1, -1):
        for j in range(4):
            if (i == 1 and j == 1):
                continue

            grid[i, j] = pi[it, 0]
            it += 1

    # display map
    fig, ax = plt.subplots()
    ax.matshow(grid, cmap=plt.cm.Blues)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i == 1 and j == 1) or (i == 1 and j == 3) or (i == 0 and j == 3):
                ax.text(j, i, "*", va='center', ha='center')
            else:
                c = grid[i, j]
                ax.text(j, i, str_actions[c], va='center', ha='center')
        
    plt.show()



def main():
    global pi

    for it in range(100):
        # evaluate current policy
        V = policy_evaluation(pi)

        # improve current policy
        pi = improve_policy(pi, V)


    show_policy()



if __name__ == "__main__":
    main()