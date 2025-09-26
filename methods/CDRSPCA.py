import sys
import os
sys.path.append('../')
sys.path.append('../datasets/synthetic')
import numpy as np
from utils import low_rank_appr
from data_gen import config, get_info, origin
import copy

def obj(S, I, Pi, i, I_c):
    return 2 * S[I_c, I][None] @ Pi[:, i][None].T - S[I_c, I_c] * Pi[i, i]

def decode(I, p):
    s = I.shape[0]
    res = np.zeros((p, s))
    for r, c in enumerate(I):
        res[c, r] = 1
    return res


def v_update(S, I, k):
    vs, es = np.linalg.eigh(S[np.ix_(I, I)])
    # In the descending order
    idx = np.argsort(np.real(-vs))[:k]
    return es[:, idx]

def idx_search(S, V, I):
    Pi = V @ V.T
    for i in range(I.shape[0]):
        r = list(set(np.arange(S.shape[0])) - set(I))
        max_v = obj(S, I, Pi, i, I[i])
        I_b = I[i]
        for j in range(len(r)):
            I_c = r[j]
            I_new = copy.deepcopy(I)
            I_new[i] = I_c
            cost = obj(S, I_new, Pi, i, I_c)
            if cost > max_v:
                I_b = I_c
                max_v = cost
        I[i] = I_b
    return I


def cdrspca(S, s, k, p, T, init_strat="random"):
    # initialization
    r = np.arange(p)
    if init_strat == "random":
        np.random.shuffle(r)
        I = r[:s]
    elif init_strat == "low rank":
        A_k = low_rank_appr(S, k)
        idx = np.argsort(-np.diag(A_k))[:s]
        I = r[idx]
    elif init_strat == "convex relaxation":
        pass
    # alternatively V and update index
    for _ in range(T):
        V = v_update(S, I, k)
        I = idx_search(S, V, I)
    
    return decode(I, p) @ V
    


if __name__ == "__main__":
    p = 128
    n = 80
    s = 5
    k = 1
    config(p, n, s, k)
    print(get_info())
    seed = np.random.randint(100)

    _, S, Sigma, Pi, U = origin(seed)

    U_h = cdrspca(S, s, k, p, 100)


    Pi_h = U_h @ U_h.T

    print(np.linalg.norm(Pi_h - Pi))
    print(np.linalg.norm(Pi_h[:s, :s] - Pi[:s, :s]))
    np.set_printoptions(precision=2)
    print(Pi[:10, :10])
    print(Pi_h[:10, :10])
