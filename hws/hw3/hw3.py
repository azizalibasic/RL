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
    return grid


def state_action_dict(grid, A=[0, 1, 2, 3]):
    state_dict = {}

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            for a in range(len(A)):
                state_dict[(r, c), a] = 0.0

    return state_dict


def state_action_dict_v3(grid, A=[0, 1, 2, 3]):
    state_dict = []

    # Left, Down, Right, Up
    for r in range(len(grid) * len(grid)):
        state_dict.append([0.0,0.0,0.0,0.0])

    return np.array(state_dict)


def best_action(Q, S):

    return np.argmax(Q[S])


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


def sarsa(amap, gamma, alpha, epsilon, episodes, seed, debug=False):
    grid = get_map(amap)
    env = gym.make('FrozenLake-v0', desc=grid).unwrapped
    np.random.seed(seed)
    env.seed(seed)
    Rs = []

    # env.render()
    #Q = state_action_dict(grid)
    Q = state_action_dict_v3(grid)
    # S = env.reset()
    if debug: print("Q:", Q)

    # print("S:",S)
    for e in range(episodes):
        # c,r
        S = env.reset()
        cnt = 0

        A = get_next_action(Q, S, epsilon)
        done = False

        if debug: print("\n\nQ:", Q)

        while not done:
            if debug: print("Q=",Q)
            if debug: print("")

            S_prime, R, done, info = env.step(A)
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


    #plt.plot(Rs)
    #plt.show()
    return Q



def run(map, gamma, alpha, epsilon, episodes, seed, debug=False):
    Q = sarsa(amap=map, gamma=gamma, alpha=alpha, epsilon=epsilon, episodes=episodes, seed=seed, debug=debug)

    bas = []
    for s in range(len(map)):
        ba = best_action(Q, s)
        bas.append(ACTION_MAP[ba])
        print("s={0},a={1}:".format(s, ACTION_MAP[ba]))

    print("PI=", ''.join(bas))

# def exp1():
#     print("============================================")
#     run(map="SFFG", gamma=1.0, alpha=0.24, epsilon=0.09, episodes=49553, seed=202404)
#     print("============================================")
#     print("\n\n")
# exp1()

# def exp2():
#     print("============================================")
#     run(map="SFFFHFFFFFFFFFFG", gamma=1.0, alpha=0.25, epsilon=0.29, episodes=14697, seed=741684)
#     print("============================================")
#     print("\n\n")
# exp2()
#
# def exp3():
#     print("============================================")
#     run(map="SFFFFHFFFFFFFFFFFFFFFFFFG", gamma=.91, alpha=0.12, epsilon=0.13, episodes=42271, seed=983459)
#     print("============================================")
#     print("\n\n")
# exp3()



# def exp1():
#     print("===================1=========================")
#     run(map="SFFG", gamma=0.92, alpha=0.16, epsilon=0.22, episodes=15817, seed=213814)
#     print("============================================")
#     print("\n\n")
# exp1()

#
# def exp2():
#     print("===================2=========================")
#     run(map="SFFFHFFFFFFFFFFG", gamma=0.93, alpha=0.29, epsilon=0.24, episodes=19261, seed=324435)
#     print("============================================")
#     print("\n\n")
# exp2()


# def exp3():
#     print("===================3=========================")
#     run(map="SFFG", gamma=0.99, alpha=0.22, epsilon=0.26, episodes=33633, seed=878811)
#     print("============================================")
#     print("\n\n")
# exp3()


# def exp4():
#     print("===================4=========================")
#     run(map="SFFFHFFFFFFFFFFG", gamma=0.97, alpha=0.18, epsilon=0.17, episodes=5981, seed=650559)
#     print("============================================")
#     print("\n\n")
# exp4()


# def exp5():
#     print("===================5=========================")
#     run(map="SFFHFFHFG", gamma=0.91, alpha=0.18, epsilon=0.11, episodes=48221, seed=392537)
#     print("============================================")
#     print("\n\n")
# exp5()
#
# def exp6():
#     print("===================6=========================")
#     run(map="SFFG", gamma=0.99, alpha=0.1, epsilon=0.22, episodes=41692, seed=106212)
#     print("============================================")
#     print("\n\n")
# exp6()

#
# def exp7():
#     print("===================7=========================")
#     run(map="SFFHFFHFG", gamma=0.96, alpha=0.09, epsilon=0.27, episodes=14082, seed=19746)
#     print("============================================")
#     print("\n\n")
# exp7()


#
# def exp8():
#     print("===================8=========================")
#     run(map="SFFHFFHFG", gamma=0.96, alpha=0.1, epsilon=0.11, episodes=7626, seed=606379)
#     print("============================================")
#     print("\n\n")
# exp8()


# def exp9():
#     print("===================8=========================")
#     run(map="SFFFHFFFFFFFFFFG", gamma=0.95, alpha=0.06, epsilon=0.11, episodes=27778, seed=277339)
#     print("============================================")
#     print("\n\n")
# exp9()
#


def exp10():
    print("===================8=========================")
    run(map="SFFG", gamma=0.95, alpha=0.26, epsilon=0.09, episodes=36413, seed=217147)
    print("============================================")
    print("\n\n")
exp10()