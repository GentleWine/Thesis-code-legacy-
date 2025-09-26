import sys
sys.path.append('../')
sys.path.append('../datasets/synthetic')
import numpy as np
from scipy.linalg import polar
from scipy.stats import ortho_group
from data_gen import config, get_info, origin

def decode(P, U):
    return np.where(P == 1, U, 0)

def pattern(N, gamma, X, T, rds):
    # For simplicity, we consider the regularization parameters to be consistent
    n = X.shape[0]
    p = X.shape[1]
    k = N.shape[0]
    Z = ortho_group.rvs(p, random_state=rds)[:, :k]
    for _ in range(T):
        for i in range(k):
            res = np.zeros((p, 1))
            for j in range(n):
                thresh = max(N[i][i] * abs(X[j][None] @ Z[:, i]) - gamma, 0)
                sign = X[j][None] @ Z[:, i] / abs(X[j][None] @ Z[:, i])
                res += sign * N[i][i] * thresh * X[j][None].T
            Z[:, i] = res.flatten()
        Z, _ = polar(Z)
    P = np.zeros((n, k))
    for i in range(k):
        for j in range(n):
            if N[i][i] * abs(X[j][None] @ Z[:, i]) > gamma:
                P[j, i] = 1
    return P

def alternate(X, P, N, T, rds):
    p = X.shape[1]
    k = N.shape[0]
    Z = ortho_group.rvs(p, random_state=rds)[:, :k]
    for _ in range(T):
        U = X @ Z @ N
        U = U @ (1 / np.sqrt(np.diag(np.diag(U.T @ U))))
        U = decode(P, U)
        Z, _ = polar(X.T @ U @ N)

    return U

def gpm(X, N, gamma, T_1, T_2, rds):
    P = pattern(N, gamma, X, T_1, rds)
    U = alternate(X, P, N, T_2, rds)
    return U

if __name__ == "__main__":
    p = 128
    n = 80
    s = 5
    k = 1
    config(p, n, s, k)
    print(get_info())
    seed = np.random.randint(100)
    _, S, Sigma, Pi, U = origin(seed)

    N = np.diag(np.arange(k, 0, -1)).astype(float)
    N /= np.linalg.norm(N)

    U_h = gpm(S, N, 1, 100, 100, rds)

    Pi_h = U_h @ U_h.T

    print(np.linalg.norm(Pi_h - Pi))
    print(np.linalg.norm(Pi_h[:s, :s] - Pi[:s, :s]))
    np.set_printoptions(precision=2)
    print(Pi[:10, :10])
    print(Pi_h[:10, :10])
