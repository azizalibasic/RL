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

def state_action_dict_v3(map_size, A):
    state_dict = []

    # Left, Down, Right, Up
    for r in range((map_size)):
        state_dict.append([0.0 for x in A])

    return np.array(state_dict)


def best_action(Q, S):

    return np.argmax(Q[S])


def get_next_action(Q, S, epsilon, A):
    p = np.random.random()
    rand = p < epsilon
    # print("SS=",S)
    if rand:
        next_action = np.random.randint(len(A))
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

def decay(a, x, r=.01):
    # y = a(1 - r)^x

    return a*(1-r)**x


def Q_LEARNER(amap, gamma, alpha, epsilon, episodes, seed, debug=False):
    ACTIONS = [0, 1, 2, 3, 4, 5]
    STATE_SIZE = 500
    #grid = state_action_dict_v3(5, A=ACTIONS)
    env = gym.make('Taxi-v3').unwrapped
    np.random.seed(seed)
    env.seed(seed)
    Rs = []

    # env.render()
    #Q = state_action_dict(grid)
    Q = state_action_dict_v3(STATE_SIZE, A=ACTIONS)
    # S = env.reset()
    if debug: print("Q:", Q)

    # print("S:",S)
    rewards = []
    for e in range(episodes):
        print("Next iter:",e)
        # c,r
        S = env.reset()
        cnt = 0

        done = False

        if debug: print("\n\nQ:", Q)

        reward = 0
        while not done:
            if debug: print("Q=",Q)
            if debug: print("")

            #Choose A from S using policy derived from Q
            A = get_next_action(Q, S, epsilon, A=ACTIONS)

            #Take action A, observe R, S_PRIME
            S_prime, R, done, info = env.step(A)
            reward += R

            next_best_action = best_action(Q, S_prime)
            Q_prime = Q[S_prime, next_best_action]
            if done:
                Q_prime = 0

            Q[S, A] = Q[S, A] + (alpha * (R + (gamma * Q_prime) - Q[S, A]))
            S = S_prime
            cnt += 1

        rewards.append(reward)
        print("\tepsilon:",epsilon)
        print("\talpha:",alpha)
        print("\tavg r:",np.average(np.array(rewards)))
        if e % 100 == 0:
            epsilon = max(.000001, epsilon * .999)
            alpha = max(.000001, alpha * .999)
            rewards = []

    #plt.plot(Rs)
    #plt.show()
    return Q



def run(map, gamma, alpha, epsilon, episodes, seed, debug=False):
    Q = Q_LEARNER(amap=map, gamma=gamma, alpha=alpha, epsilon=epsilon, episodes=episodes, seed=seed, debug=debug)

    #−11.374
    res1 = Q[462, 4]
    print("res=", res1, "--exp=", (-11.374))

    res2 = Q[398, 3]
    print("res=", res2, "--exp=", (4.348))

    res3 = Q[253, 0]
    print("res=", res3, "--exp=", (-0.585))

    res4 = Q[377, 1]
    print("res=", res4, "--exp=", (9.683))

    res5 = Q[83, 5]
    print("res=", res5, "--exp=", (-13.996))

    print('--------\n\n\n')

    items = [
        (222,2),
        (96,2),
        (11,3),
        (172,1),
        (317,4),
        (421,2),
        (321,0),
        (333,3),
        (391,3),
        (442,0)
    ]

    for i in range(len(items)):
        print("")
        x,y = items[i]
        print("i={0}, res={1}".format(i+1, Q[x,y]))

    print('--------\n\n\n')

def exp1():
    print("============================================")
    run(map="SFFG", gamma=.9, alpha=0.5, epsilon=0.5, episodes=50000, seed=202404)
    print("============================================")
    print("\n\n")
exp1()