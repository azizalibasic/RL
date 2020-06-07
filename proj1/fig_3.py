import numpy as np
import math
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt

#np.random.seed(0)

S = {
    "B": np.array([1,0,0,0,0]).T,
    "C": np.array([0,1,0,0,0]).T,
    "D": np.array([0,0,1,0,0]).T,
    "E": np.array([0,0,0,1,0]).T,
    "F": np.array([0,0,0,0,1]).T,
    "A": np.array([0,0,0,0,0]).T,
    "G": np.array([0,0,0,0,0]).T,
}


def gen_sequence():
    T = {
        "B": np.array(['A', "C"]),
        "C": np.array(["B", "D"]),
        "D": np.array(["C", "E"]),
        "E": np.array(["D", "F"]),
        "F": np.array(["E", "G"]),
    }

    S = "D"

    seqs = [S]

    while True:
        # do a random roll
        ns = T[S]
        S = np.random.choice(ns, p=[.5, .5])

        if S == "A":
            seqs.append(S)
            return (seqs, 0)
        elif S == 'G':
            seqs.append(S)
            return (seqs, 1)

        seqs.append(S)


def p(xt, w, debug=False):
    pt = np.dot(w.T, xt)
    if debug: print("\t\t\t\tpt={0}".format(pt))
    return pt

def residual(xt, l, t, k, debug=False):
    #return (l ** (t-k)) * xt
    #is it always 1?
    if debug: print("\t\t\t(t-k)={0}".format((t-k)))
    tmk = t-k
    l_tmk = l ** tmk
    res = (l_tmk) * xt
    #print("res={0}".format(res))
    return res

def residual_sum(X, t, l, debug=False):
    total = np.array([0.0,0.0,0.0,0.0,0.0])
    #t = len(X)-1
    for k in range(1, t+1):
        xk = S[X[k-1]]
        res = np.array(residual(xk, l, t, k))
        if debug: print("\t\tres={0},total={1},xk={2},lambda={3},xt={4}".format(res,total,xk,l,X[k-1]))
        total += res
    return total

def error(X,w,t,a, debug=False):
    xt = S[X[t-1]]
    xt_prime = S.get(X[t],0.0)
    pred = p(xt, w)
    pred_prime = p(xt_prime, w)

    if X[t] == 'A':
        pred_prime = 0.0
    elif X[t] == 'G':
        pred_prime = 1.0
    err = a * (pred_prime - pred)
    if debug: print("\txt={0},xt_prime={1}, pred={2}, pred_prime={3}, err={4}".format(X[t-1],X[t],pred,pred_prime,err))
    return err

def delta_w_t(X,w,t,a,l,debug=True):
    err = error(X,w,t,a)
    res = residual_sum(X, t=t, l=l)
    dw = err * res
    if debug: ("t={0},err={1},res={2},w={3},dw={4}".format(t,err,res,w,dw))
    #print("\t\tt={0},res={1}".format(t, res))
    #print("\t\tt={0},err={1}".format(t, err))
    return dw


# accumulated over sequences and
# only used to update the weight vector
# after the complete presentation of a training set.
def delta_w_t_v1(Xs, w, a, l, debug=False):
    print("alpha={0},lambda_val={1}".format(a, l))
    # Xs == training set (10 sequences)
    # seq is X (sequence)
    Ws = []
    for i in range(len(Xs)):
        X_Z = Xs[i]
        X, z = X_Z
        #print("Seq={0}".format(i))
        temp_dw = np.array([0.0,0.0,0.0,0.0,0.0])
        #print("--")
        for t in range(1, len(X)):
            temp = delta_w_t(X, w, t, a=a, l=l)
            #print("\tdw_i={0}".format(temp))
            temp_dw += temp
        Ws.append(temp_dw)
        if debug: print("t={0}, dw={1}".format(t, temp_dw))

    # return new w
    new_w = np.array(Ws)
    new_w = np.sum(new_w, axis=0)

    return new_w


def run_v1(a, l, debug=False):

    seqqq = ['D','C','D','E','F','E','F','G']

    # gen 100 training sets
    # each should have 10 sequences
    training_sets = []
    for ts in range(0, 100):
        seq = []
        for ss in range(0, 10):
            seq.append(gen_sequence())
        training_sets.append(seq)

    #training_sets = training_sets[0:3]

    # now run
    ALL_W = []
    for i in range(0,1):
        w = np.array([0.5,0.5,0.5,0.5,0.5],dtype=np.float)
        #print("finished i={0}".format(i))
        for j, train_set in enumerate(training_sets):
            for g in range(0,20):
                d_w = delta_w_t_v1(train_set, w, a, l, debug=False)

                w += d_w
                if debug: print("train_set_i={0}, old_w={1}, new_w={2}".format(j, w-d_w, w))
                #if debug: print("\td_w={0}".format(d_w))
                #if debug: print("")
            ALL_W.append(w)

    new_w = np.array(ALL_W)
    new_w = np.average(new_w, axis=0)
    return new_w

###### RUNNING STUFF #######
#run_v1(a=0.01,l=.2, debug=True)
#un_v1(a=.001,l=0.0, debug=True)


def main3():
    np.random.seed(0)
    EXPECTED = [1/6., 1/3., 1/2., 2/3., 5/6.]
    ALPHA = 0.0025
    w0 = run_v1(a=ALPHA, l=0.0, debug=True)
    w1 = run_v1(a=ALPHA, l=0.1, debug=True)
    w3 = run_v1(a=ALPHA, l=0.3, debug=True)
    w5 = run_v1(a=ALPHA, l=0.5, debug=True)
    w7 = run_v1(a=ALPHA, l=0.7, debug=True)
    w10 = run_v1(a=ALPHA, l=1.0, debug=True)

    ws = [w0, w1, w3, w5, w7, w10]

    errors = []

    for i in range(len(ws)):
        rmse = sqrt(mean_squared_error(EXPECTED, ws[i]))
        errors.append(rmse)

    plt.plot([0,.1,.3,.5,.7,1], errors)
    plt.xlabel('Lambda(s)')
    plt.ylabel('Errors (RMSE)')
    plt.title('Figure 3')
    plt.savefig('figure3.png')
    plt.clf()

# if __name__ == "__main__":
#     main3()


