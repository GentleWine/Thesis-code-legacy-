import numpy as np
import sys
sys.path.append('../')
sys.path.append('../datasets/synthetic')
from data_gen import config, get_info, origin
from utils import gen_orth_mat

def truncate(U, s):
    v = np.linalg.norm(U, axis=1)
    ex = np.argsort(-v)
    ex = ex[-1:s:-1]
    U[ex] = 0
    return U

def stream(X, T, B, s, k):
    p = X.shape[1]
    Q = np.zeros((p, k))
    for t1 in range(T):
        S_t = np.zeros((p, k))
        for t2 in range(B * t1 + 1, B * (t1 + 1) + 1):
            S_t = S_t + 1 / B * X[t2][None].T @ X[t2][None] @ Q
        S = truncate(S_t, s)
        Q, _ = np.linalg.qr(S)
    return Q

if __name__ == "__main__":
    p = 128
    n = 80
    s = 5
    k = 1
    config(p, n, s, k)
    print(get_info())
    seed = np.random.randint(100)
    X, S, Sigma, Pi, U = origin(seed)
    U_h = stream(X, 8, 9, s, k)
    
    Pi_h = U_h @ U_h.T

    print(np.linalg.norm(Pi_h - Pi))
    print(np.linalg.norm(Pi_h[:s, :s] - Pi[:s, :s]))
    np.set_printoptions(precision=1)
    print(Pi[:10, :10])
    print(Pi_h[:10, :10])
