import sys
sys.path.append("../")
import numpy as np
from utils import gen_orth_mat
### default setting
# the scale of the problem
p = 128
# the number of samples
n = 80
# the sparsity
s = 10
# the dimension of eigenspace
k = 5

def config(new_p, new_n, new_s, new_k):
    global p
    global n
    global s
    global k
    
    p = new_p
    n = new_n
    s = new_s
    k = new_k

def get_info():
    return p, n, s, k
def origin(seed):
    rds = np.random.RandomState(seed)
    U = gen_orth_mat(p, s, rds, arti=False)
    np.random.seed(seed)
    # projector
    Pi = U[:,:k] @ U[:,:k].T

    # eigenvalues
    es = np.ones(p)
    for i in range(k):
        if i == k - 1:
            es[i] = 10
        else:
            es[i] = 100

    E = np.diag(es)

    # the covariance matrix
    Sigma = U @ E @ U.T


    # step2: sampling
    Xs = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)

    # calculate the estimated covariance matrix
    Sigma_h = Xs.T @ Xs / n

    return Xs, Sigma_h, Sigma, Pi, U
