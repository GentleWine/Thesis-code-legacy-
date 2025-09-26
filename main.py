import numpy as np
from scipy.stats import ortho_group
from algs import *
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

# student-t distribution
from scipy.stats import multivariate_normal, chi2

def multivariate_t_rvs(mu, cov, df, size=1):
    """
    生成多元 Student-t 分布的随机样本
    :param mu: 均值向量
    :param sigma: 协方差矩阵
    :param df: 自由度
    :param size: 采样数量
    :return: 采样数据 (size, d)
    """
    d = len(mu)  # 维度
    g = np.tile(chi2.rvs(df, size=size) / df, (d, 1)).T  # 生成卡方变量
    z = multivariate_normal.rvs(mean=np.zeros(d), cov=cov, size=size)  # 生成正态变量
    return mu + z / np.sqrt(g)



# the scale of the problem
p = 100
# the number of samples
n = 60
# the sparsity
s = 10
# the dimension of eigenspace
k = 5

seed = 0

### covariance case 1: normal covariance model
H_SNR = False

# create the covariance matrix
U, supp = gen_orth_mat(p, s, k, 1)
assert np.linalg.matrix_rank(U) == p
Pi = U[:,:k] @ U[:,:k].T
if H_SNR:
    es = 0.1 * np.ones(p)
    for i in range(k):
        if i == k - 1:
            es[i] = 100
        else:
            es[i] = 1000
else:
    es = np.ones(p)
    for i in range(k):
        if i == k - 1:
            es[i] = 10
        else:
            es[i] = 100

E = np.diag(es)
Sigma = U @ E @ U.T


### covariance case 2: spiked covariance model
# U, supp = gen_orth_mat(p, s, k, seed)
# assert np.linalg.matrix_rank(U) == p
# Pi = U[:,:k] @ U[:,:k].T

# sigma = 1.0
# theta = [10, 10, 10, 10, 10]
# Sigma = sigma ** 2 * np.eye(p)
# for i in range(k):
#     Sigma += theta[i] * np.outer(U[:, i], U[:, i])


### distribution case 1: normal distribution
# # random seed
# np.random.seed(0)
# # step2: sampling
# Xs = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)

### distribution case 2: t-distribution
# covariance matrix conversion
df = 3
cov = (df - 2) / df * Sigma
# random seed
np.random.seed(seed)
# sampling
Xs = multivariate_t_rvs(np.zeros(p), cov, df, n)


# calculate the estimated covariance matrix
Sigma_h = Xs.T @ Xs / n

### case 1: normal covariance model
# Pi_h, hist = mm(Sigma_h, 1, 0.7, 3.625, k, 20, 10, 10)
# Pi_h, hist = Gu(Sigma_h, k, 0, 9, 0.5005, 3.625, 50, p, Pi, strategy="mcp")
# Pi_h, _ = Wang(Sigma_h, k, 1, 10, 0.5005, 1, p, 1.2*s, Pi, 80, 20)
# Pi_h= PCA(Sigma_h, k)

### case 2: spiked covariance model
# Pi_h, hist = mm(Sigma_h, 1, 0.5005, 3.625, k, 20, 10, 10)
# Pi_h, hist = Gu(Sigma_h, k, 0, 9, 0.5005, 3.625, 50, p, Pi, strategy="mcp")
# Pi_h, _ = Wang(Sigma_h, k, 1, 10, 0.5005, 1, p, 1.2*s, Pi, 80, 20)
# Pi_h= PCA(Sigma_h, k)

### case 3: t-distribution
Pi_h, hist = mm(Sigma_h, 1, 3.5005, 3.625, k, 20, 10, 10)
# Pi_h, hist = Gu(Sigma_h, k, 0, 9, 2.5005, 3.625, 50, p, Pi, strategy="mcp")
# Pi_h, _ = Wang(Sigma_h, k, 1, 10, 0.5005, 1, p, 1.2*s, Pi, 80, 20)
# Pi_h, _ = Wang(Sigma_h, k, 1, 10, 0.5005, 1, p, 1.5*s, Pi, 80, 20)
# Pi_h, _ = Wang(Sigma_h, k, 1, 10, 0.5005, 1, p, 2*s, Pi, 80, 20)
# Pi_h= PCA(Sigma_h, k)


np.set_printoptions(precision=2, suppress=True)

# print(Pi[np.ix_(supp, supp)])
# print(Pi_h[np.ix_(supp, supp)])

print(np.linalg.norm(Pi_h - Pi))

tp = 0
fp = 0
fn = 0
tn = 0
for i in range(p):
    for j in range(p):
        if i in supp and j in supp:
            if np.abs(Pi_h[i][j]) > 1e-10:
                tp += 1
            else:
                fn += 1
        else:
            if np.abs(Pi_h[i][j]) > 1e-10:
                fp += 1
            else:
                tn += 1
print("TPR:", tp / (tp + fn))
print("FPR:", fp / (fp + tn))
