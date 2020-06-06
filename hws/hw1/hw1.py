import math

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def reward(s, a, s_prime, is_bad_side):
    if s_prime == 'sl':
        return 0.0
    elif s == 'sl':
        return 0.0
    elif s == 'sw' and s_prime == 'sw':
        return 0.0
    elif s == 'ss' and s_prime == 'ss':
        return 0.0
    elif s == 'ss' and s_prime == 'sw':
        return 0.0
    elif s_prime == 'sw':
        current_val = int(s[1:])
        return current_val
    else:
        return 0.0

def reward_v2(s, a, s_prime, is_bad_side):
    next_val = try_parse_state_num(s_prime)
    current_val = try_parse_state_num(s)

    # if at end or lose r = 0
    if s == 'sl':
        return 0.0
    elif s == 'sw':
        return 0.0
    # going to end or lose r = 0
    if s_prime == 'sw':
        return 0.0
    elif s_prime == 'ss':
        return 0.0

    if s == 'ss' and s_prime == 'sl':
        return 0.0
    elif s == 'ss':
        return next_val

    if s_prime == 'sl':
        return -current_val

    # r = sprime_r - s_r
    return next_val - current_val


def try_parse_state_num(s):
    try:
        current_val = int(s[1:])
        return current_val
    except:
        return -1

def is_bad_num(n, is_bad_side):
    if n > len(is_bad_side):
        return False
    if is_bad_side[n-1] == 0:
        return False

    return True


def state_reachable(s, s_prime, is_bad_side):
    s_prime_num = try_parse_state_num(s_prime)

    if s_prime_num == -1:
        return False

    if not is_bad_num(s_prime_num, is_bad_side):
        return True

    return False


def win_lose_prob(is_bad_side):
    n = len(is_bad_side)
    num_ones = len([x for x in is_bad_side if x == 1])
    num_zeros = len([x for x in is_bad_side if x == 0])
    l_p = float(num_ones / n)
    w_p = (1.0-l_p)/num_zeros

    return w_p, l_p


def transition_prob(s, action, s_prime, is_bad_side, possible_states):
    # Stay at lose
    if s == 'sl' and s_prime == 'sl':
        return 1.0
    elif s == 'sl' and s_prime != 'sl':
        return 0.0
    # Stay at win
    if s == 'sw' and s_prime == 'sw':
        return 1.0
    elif s == 'sw' and s_prime != 'sw':
        return 0.0

    w_p, l_p = win_lose_prob(is_bad_side)

    if action == 't':
        #possible = state_reachable(s, s_prime, is_bad_side)
        if s_prime == 'sl':
            return l_p
        else:
            return w_p
    elif action == 's':
        # Stop transition is 1 if you are moving to win
        if s_prime == 'sw':
            return 1.0
        else:
            return 0.0

    print(
        "UNDEFINED. s={0}, action={1}, s_prime={2}, next_state_nums={3}"
            .format(s, action, s_prime, is_bad_side))
    raise Exception("undefined behavior")


def next_states(s, a, is_bad_side):

    if s == 'sl':
        return ['sl']
    if s == 'sw':
        return ['sw']

    if a == 's':
        return ['sw']

    new_states = []
    for i, gb_s in enumerate(is_bad_side):
        if gb_s == 0:
            if s == 'ss':
                num = 0
            else:
                num = try_parse_state_num(s)

            val = num+i+1
            #if not is_bad_num(val, is_bad_side):
            new_states.append("s"+str(val))

    return new_states

def state_to_state(s, s_prime, a):

    next_val = try_parse_state_num(s_prime)
    current_val = try_parse_state_num(s)

    if a == 't' and  s == 'ss' and next_val > 0:
        return True
    elif a == 't' and current_val > 0 and next_val > 0:
        return True

    return False

def val_iter(A, Vs, is_bad_side, epsilon=0.001, max_iters=10, min_iters=10):
    print("Start vs:", Vs)
    i = 0
    S = list(Vs.keys())
    while True:
        i+=1
        if i > max_iters:
            print("New vs:", Vs)
            return Vs
        d = 0
        print("======")
        #S = list(Vs.keys())
        for s in S:
            v = Vs[s]
            for a in A:
                temp = 0
                print("--")
                possible_states = next_states(s, a, is_bad_side)
                if a == 's':
                    possible_states = ['sw']
                else:
                    possible_states.append('sl')

                for s_prime in possible_states:
                    t = transition_prob(s, a, s_prime, is_bad_side, possible_states)

                    #reward only on sw,sl
                    #r = reward(s, a, s_prime, is_bad_side)
                    r = reward_v2(s, a, s_prime, is_bad_side)
                    if s_prime not in Vs:
                        Vs[s_prime] = 0.0
                    vs_prime = Vs[s_prime]
                    temp += (t * (r + vs_prime))

                    print(
                        "s={0}, s'={1}, ex={2}, t={3}, r={4}, vs_prime={5}, a={6}"
                        .format(s,s_prime,temp,t,r,vs_prime,a)
                    )
                if temp > Vs[s]:
                    Vs[s] = temp
            d = max(d, abs(v - Vs[s]))

        print("New vs:", Vs)
        if d < epsilon:
            print("Exiting=d={0}, e={1}, iter_count={2}".format(d,epsilon, i))
            print("New vs:", Vs)
            return Vs


def run_game(is_bad_side):
    S = []
    for i in range(len(is_bad_side) * 3):
        S.append('s'+(str(i+1)))

    S.append('sl')
    S.append('sw')
    S.append('ss')

    A = ['t', 's']

    Vs = {}
    for k in S:
        Vs[k] = 0.0

    new_vs = val_iter(A, Vs, is_bad_side, epsilon=0.001, max_iters=200)
    return new_vs['ss'], new_vs


# GAME 2
# s0, Vs = run_game(is_bad_side=[1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0])
# print("S0=",s0)
# assert truncate(s0, 3) == 7.379


# GAME 1
# s0, Vs = run_game(is_bad_side=[1,1,1,0,0,0])
# print("S0=",s0)
# assert truncate(s0, 3) == 2.583


# GAME 3
# s0, Vs = run_game(is_bad_side=[1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0])
# print("S0=",s0)
# assert truncate(s0, 3) == 6.314

########SUBMISSION

# GAME 11
print("--------")
s0, Vs = run_game(is_bad_side=[0,0,0,0,1,1,1,1,1,1,0,0,0,0,1])
print("S0=",truncate(s0, 3))
print("--------\n\n")


# GAME 12
print("--------")
s0, Vs = run_game(is_bad_side=[0,1,0,0,0,1,0,1,0,0,1,0,0])
print("S0=",truncate(s0, 3))
print("--------\n\n")

# GAME 13
print("--------")
s0, Vs = run_game(is_bad_side=[0,0,1,0,1,0,1,1,0])
print("S0=",truncate(s0, 3))
print("--------\n\n")


# GAME 14
print("--------")
s0, Vs = run_game(is_bad_side=[0,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0])
print("S0=",truncate(s0, 3))
print("--------\n\n")


# GAME 15
print("--------")
s0, Vs = run_game(is_bad_side=[0,1,0])
print("S0=",truncate(s0, 3))
print("--------\n\n")


# GAME 16
print("--------")
s0, Vs = run_game(is_bad_side=[0,1,1,1,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,1,0,1,1,1,1])
print("S0=",truncate(s0, 3))
print("--------\n\n")


# GAME 17
print("--------")
s0, Vs = run_game(is_bad_side=[0,0,1,0,1,0,1,1,0,1,1,1,1,1,0,1,0])
print("S0=",truncate(s0, 3))
print("--------\n\n")


# GAME 18
print("--------")
s0, Vs = run_game(is_bad_side=[0,0,1,0,0,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0])
print("S0=",truncate(s0, 3))
print("--------\n\n")


# GAME 19
print("--------")
s0, Vs = run_game(is_bad_side=[0,0,0,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1])
print("S0=",truncate(s0, 3))
print("S0=",s0)
print("--------\n\n")


# GAME 20
print("--------")
s0, Vs = run_game(is_bad_side=[0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,0,0,1,0])
print("S0=",truncate(s0, 3))
print("S0=",s0)
print("--------\n\n")