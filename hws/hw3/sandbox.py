import numpy as np
import gym
import math
import matplotlib.pyplot as plt

LEFT = '<'
DOWN = 'v'
RIGHT = '>'
UP = '^'

#"Left","Down","Right","Up"

ACTION_MAP = {
    0:LEFT,
    1:DOWN,
    2:RIGHT,
    3:UP
}

def get_map(map_str):
    #amap="SFFG"
    sqr = int(math.sqrt(len(map_str)))
    grid = []
    cnt = 0
    for i in range(sqr):
        inner_grid = []
        for j in range(sqr):
            inner_grid.append(map_str[cnt])
            cnt+=1
        grid.append(inner_grid)
    #print("Sqr:", sqr)
    #print("grid:", grid)
    return grid


def state_action_dict(grid, A=[0, 1, 2, 3]):
    state_dict = {}

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            for a in range(len(A)):
                state_dict[(r, c), a] = 0.0

    return state_dict


def state_action_dict_v2(grid, A=[0, 1, 2, 3]):
    state_dict = {}

    cnt = 0
    for r in range(2*len(grid)):
        for a in range(len(A)):
            state_dict[cnt, a] = 0

        cnt += 1

    return state_dict


def best_action(Q, S, A=[0, 1, 2, 3]):
    next_action = -10000.0
    next_action_score = -10000.0
    for a in A:
        # print("S,a", S,a)
        na_score = Q[S, a]
        if na_score > next_action_score:
            next_action_score = na_score
            next_action = a

    return next_action


def get_next_action(Q, S, epsilon, A=[0, 1, 2, 3]):
    p = np.random.random()
    rand = p < epsilon
    # print("SS=",S)
    if rand:
        next_action = np.random.randint(4)
    else:
        next_action = best_action(Q, S)


    return next_action

def state_to_r_c(S, grid):
    sqr = len(grid)
    cnt = 0
    for r in range(sqr):
        for c in range(sqr):
            if cnt == S:
                return (r,c)
            cnt+=1


def sarsa(amap, gamma, alpha, epsilon, episodes, seed):
    np.random.seed(seed)
    grid = get_map(amap)
    env = gym.make('FrozenLake-v0', desc=grid, is_slippery=False).unwrapped
    env.seed(seed)
    Rs = []

    # env.render()
    #Q = state_action_dict(grid)
    Q = state_action_dict_v2(grid)
    # S = env.reset()
    print("Q:", Q)

    # print("S:",S)
    for e in range(episodes):
        # c,r
        # print("episode:", e)
        S = env.reset()
        #S = state_to_r_c(S, grid)
        cnt = 0

        A = get_next_action(Q, S, epsilon)
        done = False

        print("\n\nQ:", Q)

        z = 0
        while not done:
            print("z=",z)
            print("Q=",Q)
            print("")
            z+=1
            if z == 100:
                pass
                #return
            S_prime, R, done, info = env.step(A)
            #S_prime = state_to_r_c(S_prime, grid)
            A_prime = get_next_action(Q, S_prime, epsilon)

            Q[S, A] = Q[S, A] + (alpha * (R + (gamma * Q[S_prime, A_prime]) - Q[S, A]))
            S = S_prime
            A = A_prime
            cnt += 1
            if cnt % 400 == 0:
                print("e={0},cnt={1}".format(e, cnt))
                env.render()

            if done:
                Rs.append(R)


    plt.plot(Rs)
    plt.show()
    return Q




Q = sarsa("SFFG", gamma=1.0, alpha=0.24, epsilon=0.09, episodes=49553, seed=202404)
print("\n\nQ:", Q)

for s in range(4):
    ba = best_action(Q, 0)
    print("s={0},a={1}:".format(s, ACTION_MAP[ba]))