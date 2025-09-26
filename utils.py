import numpy as np
from scipy.stats import ortho_group
from json import JSONEncoder

def gen_orth_mat(p, s, k, seed):
    """
    Randomly generate a p * p orthogonormal matrix with the first k columns has sparsity s
    """
    rds = np.random.RandomState(seed=seed)
    U1 = ortho_group.rvs(s, random_state=rds)[:,:k]
    insert_indices = np.sort(rds.choice(p, p - s, replace=False))
    U1_zero = []
    orig_pointer = 0
    for i in range(p):
        if i in insert_indices:
            # Insert a zero row
            U1_zero.append(np.zeros(k))
        else:
            # Insert the original row
            U1_zero.append(U1[orig_pointer])
            orig_pointer += 1

    U2 = np.array(U1_zero)

    U2_comp = rds.randn(p, p - k)
    U2_comp = U2_comp - U2 @ U2.T @ U2_comp
    U2_comp, _ = np.linalg.qr(U2_comp)
    U3 = np.hstack((U2, U2_comp))
    supp = np.setdiff1d(np.arange(p), insert_indices)
    return U3, supp

def mcp_fun(X, beta, lamb):
    def fun(x, beta, lamb):
        if abs(x) <= beta * lamb:
            return lamb * abs(x) - x ** 2 / (2 * beta)
        else:
            return lamb ** 2 * beta / 2
    v_fun = np.vectorize(fun)
    T = v_fun(X, beta, lamb)
    return np.trace(T)

def get_TPRs(Pi, Pi_h, s):
    tp = 0
    fn = 0
    for i in range(s):
        for j in range(s):
            if np.abs(Pi_h[i][j]) == 0:
                fn += 1
            else:
                tp += 1
    return tp / (tp + fn)

def get_FPRs(Pi, Pi_h, s):
    p = Pi.shape[0]
    fp = 0
    tn = 0
    for i in range(s, p):
        for j in range(s, p):
            if np.abs(Pi_h[i][j]) == 0:
                tn += 1
            else:
                fp += 1
    return fp / (fp + tn)

def low_rank_appr(S, k):
    vs, es = np.linalg.eigh(S)
    # In the descending order
    idx = np.argsort(np.real(-vs))[:k]
    vs = vs[idx]
    es = es[:, idx]

    return es @ np.diag(vs) @ es.T
    

def mm_criterion(S, Pi_b, Psi_b, Z_b, L, k):

    evalue, _ = np.linalg.eigh(S + Z_b)
    idx = np.argsort(np.real(-evalue))
    evalue = evalue[idx]
    I = sum(evalue[:k])
    II = - np.trace(S @ Pi_b)
    III = np.sum(np.abs(np.multiply(L, Psi_b)))
    IV = np.sum(np.abs(np.multiply(L, Pi_b - Psi_b)))
    return I + II + III + IV

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    

if __name__ == "__main__":
    A = np.ones((3, 3))
    print(A)
    print(low_rank_appr(A, 1))

