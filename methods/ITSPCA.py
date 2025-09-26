import numpy as np
import sys
sys.path.append('../')
sys.path.append('../datasets/synthetic')
from data_gen import config, get_info, origin

def get_var(X):
    p = X.shape[1]
    x = np.mean(X ** 2, axis=0)
    if p % 2 == 0:
        var_h = (x[int(p/2-1)] + x[int(p/2)]) / 2
    else:
        var_h = x[int((p+1)/2)]
    return var_h


def dtspca(X, k, alpha):
    var_h = get_var(X)
    n = X.shape[0]
    p = X.shape[1]
    S = 1 / n * X.T @ X
    B = np.where(np.diag(S) >= var_h * (1 + alpha), 1, 0)
    keep_indices = np.array(B == 1).nonzero()[0]
    sub = X[np.ix_(keep_indices, keep_indices)]
    _, qs = np.linalg.eigh(sub)
    s_h = sub.shape[0]
    if k > s_h:
        raise("wrong estimation of s_h!")
    Q = np.zeros((p, s_h))
    for idx, row in zip(keep_indices, qs):
        Q[idx] = row
    return Q
        

def hard(M, gamma):
    return np.where(M > gamma, M, 0)

def soft(M, gamma):
    return np.multiply(np.sign(M), np.maximum(np.abs(M) - gamma,0))

def itspca(X, k, gamma, alpha, T, strategy="soft"):
    # for simplicity, we assume gamma to be consistent
    n = X.shape[0]
    S = 1 / n * X.T @ X
    Q = dtspca(X, k, alpha)[:, :k]
    for _ in range(T):
        M = S @ Q
        if strategy == "soft":
            M = soft(M, gamma)
        elif strategy == "hard":
            M = hard(M, gamma)
        Q, _ = np.linalg.qr(M)
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
    
    U_h = itspca(X, k, 0.1, 0.5, 100, strategy="soft")

    Pi_h = U_h @ U_h.T

    print(np.linalg.norm(Pi_h - Pi))
    print(np.linalg.norm(Pi_h[:s, :s] - Pi[:s, :s]))
    np.set_printoptions(precision=1)
    print(Pi[:10, :10])
    print(Pi_h[:10, :10])
    