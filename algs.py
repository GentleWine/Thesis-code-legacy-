import numpy as np
import cvxpy as cp
from aux import *
from projectors import projector_fan, operator_l1, operator_mcp, operator_sts
from utils import *
from tqdm import tqdm
def mm(S, scale, lamb, alpha, k, rho, T, T1, strat="mcp", crit=False):
    S = S / scale
    p = S.shape[0]

    Pi_h = np.zeros((p, p))
    Psi = np.zeros((p, p))
    Z = np.zeros((p, p))
    
    hist = [Pi_h]

    # record the history of stop criterion
    crit_hist = [[] for _ in range(T)]
    for t in tqdm(range(T)):
        if strat == "mcp":
            L = mcp(np.abs(Pi_h), lamb, alpha)
        elif strat == "scad":
            L = scad(np.abs(Pi_h), lamb, alpha)
        
        # np.set_printoptions(precision=3, suppress=True)
        # print(L[:20, :20])
        # print(Pi_h[:20, :20])

        # np.set_printoptions(precision=3, suppress=True)
        # print("abs(Pi_h): ", np.abs(Pi_h)[:10, :10])
        # print("L: ", L[:50, :50])

        Pi_h_sum = np.zeros((p, p))
        Psi_sum = np.zeros((p, p))
        Z_sum = np.zeros((p, p))
        for t1 in range(T1):
            
            # print("loss of obj %f at %i iters" % (-np.trace(S.T @ Pi_h) + np.sum(np.abs(L * Pi_h)), t))
        
            Pi_h = projector_fan(Psi + (Z + S) / rho, k)
            
            Psi = operator_sts(Pi_h - 1 / rho * Z, L / rho)

            Z = Z - rho * (Pi_h - Psi)

            Pi_h_sum += Pi_h
            Psi_sum += Psi
            Z_sum += Z

            # print the stop criterion

            if crit:
                np.set_printoptions(precision=3, suppress=True)
                c = mm_criterion(S, Pi_h_sum / (t1 + 1), Psi_sum / (t1 + 1), Z_sum / (t1 + 1), L, k)
                # print("criterion of iteration %i: %f" % (t1, c))

                crit_hist[t].append(c)

        Pi_h = Pi_h_sum / T1
        hist.append(Pi_h)
    
    # restore the projector
    evalue, evector = np.linalg.eigh(Pi_h)
    
    # # judge whether oracle
    # print("evalues:", evalue)

    # restore the projector
    idx = np.argsort(np.real(-evalue))
    evector = evector[:, idx]
    proj = evector[:, :k] @ evector[:, :k].T
    
    if crit:
        return proj, hist, crit_hist
    else:
        return proj, hist


def mm_orth(S, lamb, alpha, k, rho, T, T1=int(1e6), strat="mcp"):
    # output orthogonal projector in each layer
    p = S.shape[0]
    Pi_h = np.zeros((p, p))
    hist = [Pi_h]
    for _ in tqdm(range(T)):
        if strat == "mcp":
            L = mcp(np.abs(Pi_h), lamb, alpha)
        elif strat == "scad":
            L = scad(np.abs(Pi_h), lamb, alpha)

        Psi = np.zeros((p, p))
        Z = np.zeros((p, p))

        Pi_h_sum = np.zeros((p, p))
        Psi_sum = np.zeros((p, p))
        Z_sum = np.zeros((p, p))
        for t in range(T1):
            
            Pi_h = projector_fan(Psi + (Z + S) / rho, k)
            
            Psi = operator_sts(Pi_h - 1 / rho * Z, L / rho)

            Z = Z - rho * (Pi_h - Psi)

            Pi_h_sum += Pi_h
            Psi_sum += Psi
            Z_sum += Z

        Pi_h = Psi_sum / T1

        # restore the projector
        evalue, evector = np.linalg.eigh(Pi_h)
        idx = np.argsort(np.real(-evalue))
        evector = evector[:, idx]
        Pi_h = evector[:, :k] @ evector[:, :k].T

        hist.append(Pi_h)
    
    return Pi_h, hist


def Gu(S, k, tau, rho, lam, alpha, T, p, Pi, strategy):
    # initialization
    Pi_h = np.zeros((p, p))
    Phi = np.zeros((p, p))
    Theta = np.zeros((p, p))
    # history of error
    err_hist = [np.linalg.norm(Pi - Pi_h)]
    for t in tqdm(range(T)):
        
        # print("loss of obj %f at %i iters" % (-np.trace(S.T @ Pi_h) + tau / 2 * np.trace(Pi_h.T @ Pi_h) + mcp_fun(Pi_h, beta, lam), t))

        # update for Pi_h
        X = rho / (rho + tau) * Phi - 1 / (rho + tau) * Theta + 1 / (rho + tau) * S
        Pi_h = projector_fan(X, k)
        # update for Phi
        if strategy == "l1":
            Phi = operator_l1(Pi_h + 1 / rho * Theta, lam / rho)
        elif strategy == "mcp":
            Phi = operator_mcp(Pi_h + 1 / rho * Theta, lam / rho, alpha)

        # update for Theta
        Theta = Theta + rho * (Pi_h - Phi)
        # update the history
        err_hist.append(np.linalg.norm(Pi - Pi_h))

    # restore the projector
    evalue, evector = np.linalg.eigh(Pi_h)
    
    # # judge whether oracle
    # print("evalues:", evalue)

    # restore the projector
    idx = np.argsort(np.real(-evalue))
    evector = evector[:, idx]
    proj = evector[:, :k] @ evector[:, :k].T    

    return proj, err_hist


def Wang(S, k, rho, tau, lam, scale, p, s, Pi, t1, t2):
    # initialization stage
    Pi_h = np.zeros((p, p))
    Phi = np.zeros((p, p))
    Theta = np.zeros((p, p))
    Pi_h_hist = []
    # history of error
    err_hist = [np.linalg.norm(Pi - Pi_h)]
    for _ in tqdm(range(t1)):
        Pi_h = projector_fan(rho / (rho + tau) * Phi - 1 / (rho + tau) * Theta + 1 / (rho + tau) * S, k)
        Phi = operator_l1(Pi_h + 1 / rho * Theta, lam / rho)
        Theta = Theta + rho * (Pi_h - Phi)
        Pi_h_hist.append(Pi_h)
        err_hist.append(np.linalg.norm(Pi - Pi_h))
    
    Pi_h_init = np.mean(Pi_h_hist, 0)
    evalue, evector = np.linalg.eigh(Pi_h_init)
    idx = np.argsort(np.real(-evalue))
    U = evector[:, idx[:k]]

    U, _ = np.linalg.qr(_trunc(U, int(scale * s)))
    for _ in tqdm(range(t2)):
        V, _ = np.linalg.qr(S @ U)
        U, _ = np.linalg.qr(_trunc(V, int(scale * s)))
        err_hist.append(np.linalg.norm(Pi - U @ U.T))
    
    return U @ U.T, err_hist



def _trunc(V, s):
    idx = np.argsort(-np.linalg.norm(V, axis=1))
    res = np.zeros(V.shape)
    res[idx[:s]] = V[idx[:s]]
    return res


def primal_dual(S, lamb, alpha, k, T, strat="mcp"):
    """
    primal dual algorithm for nonconvex penalized PCA formulation.
    solving by cvx.
    """
    p = S.shape[0]
    Pi_h = np.zeros((p, p))
    for _ in tqdm(range(T)):
        if strat == "mcp":
            L = mcp(np.abs(Pi_h), lamb, alpha)
        elif strat == "scad":
            L = scad(np.abs(Pi_h), lamb, alpha)

        # solving by cvx
        X = cp.Variable((p, p), symmetric=True)
        constraints = [X >> 0]
        constraints += [np.eye(p) >> X]
        constraints += [cp.trace(X) == k]
        prob = cp.Problem(cp.Minimize(-cp.trace(S @ X) + cp.atoms.norm1(cp.multiply(L, X))), constraints)
        prob.solve()

        # update the variable
        Pi_h = X.value
    return Pi_h

def oracle(S, k, supp):
    p = S.shape[0]
    mask = np.zeros_like(S, dtype=bool)
    mask[np.ix_(supp, supp)] = True
    S_supp = np.where(mask, S, 0)
    
    # solving by cvx
    X = cp.Variable((p, p), symmetric=True)
    constraints = [X >> 0]
    constraints += [np.eye(p) >> X]
    constraints += [cp.trace(X) == k]
    prob = cp.Problem(cp.Minimize(-cp.trace(S_supp @ X)), constraints)
    prob.solve()

    # update the variable
    Pi_h = X.value

    return Pi_h

def PCA(S, k):
    evalue, evector = np.linalg.eigh(S)
    idx = np.argsort(np.real(-evalue))
    evector = evector[:, idx]
    return evector[:, :k] @ evector[:, :k].T
