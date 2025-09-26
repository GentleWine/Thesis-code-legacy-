import numpy as np
import cupy as cp
def projector_fan(X, k):
    """
    Fantope projector
    """
    p = X.shape[0]

    # validation
    if k > p:
        return False
    if p > 700:
        evalue, evector = cp.linalg.eigh(cp.asarray(X))
        evalue = cp.asnumpy(evalue)
        evector = cp.asnumpy(evector)
    else:
        evalue, evector = np.linalg.eigh(X)

    # In the descending order
    idx = np.argsort(np.real(-evalue))
    evalue = evalue[idx]
    evector = evector[:, idx]

    # extended eigenvalues
    ext_evalue = evalue.tolist()
    ext_evalue.append(-1e10)
    # new eigen values
    newvalue = np.zeros(p)

    # summation counter
    s = k

    # head and tail of the processing sequence
    i = 0
    j = 1

    head_dist = 1
    tail_dist = ext_evalue[0] - ext_evalue[1]

    flag = False
    # calculate new eigenvalues incremetally,
    # in each step, proceed one unit
    while j <= p:
        vul = j - i
        r = min(head_dist, tail_dist)
        if s <= r * vul:
            r = s / vul
            flag = True
        # proceeding
        for m in range(i, j):
            newvalue[m] += r
            s -= r

        if flag:
            break
        else:
            if head_dist > tail_dist:
                head_dist -= tail_dist
                tail_dist = ext_evalue[j] - ext_evalue[j+1]
                j += 1
            else:
                tail_dist -= head_dist
                if j - i <= 1:
                    if j < p:
                        tail_dist = ext_evalue[j] - ext_evalue[j+1]
                    j += 1
                head_dist = min(ext_evalue[i] - ext_evalue[i+1], 1)
                i += 1
    if j > p:
        return False
    
    E = np.diag(newvalue)
    return evector @ E @ evector.T


# def orth_projector(X, k):
#     """
#     orthogonal projector
#     """
#     p = X.shape[0]

#     # validation
#     if k > p:
#         return False
#     if p > 700:
#         evalue, evector = cp.linalg.eigh(cp.asarray(X))
#         evalue = cp.asnumpy(evalue)
#         evector = cp.asnumpy(evector)
#     else:
#         evalue, evector = np.linalg.eigh(X)

#     # In the descending order
#     idx = np.argsort(np.real(-evalue))
#     evector = evector[:, idx][:k]
    
#     return evector @ evector.T


def operator_l1(X, lam):
    """
    soft-thresholding operator
    """
    n = X.shape[0]
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = X[i, j]
            if x <= -lam:
                res[i, j] = x + lam
            elif x > lam:
                res[i, j] = x - lam
    return res

def operator_mcp(X, lamb, alpha):
    """
    soft-thresholding operator
    """
    n = X.shape[0]
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = X[i, j]
            if abs(x) <= lamb * alpha:
                s = 0
                if x > lamb:
                    s = x - lamb
                if x < -lamb:
                    s = x + lamb
                res[i, j] = s / (1 - 1 / alpha)
            else:
                res[i, j] = x
    return res

def operator_sts(X, L):
    return np.multiply(np.sign(X), np.maximum(np.abs(X) - L, 0))
