# conduct multiple experiments
import numpy as np
from utils import *
from algs import *
import os
import json
import datetime
import time
def grid_mm(rep, k, s, p, n, path, strat="mcp"):
    lambs = np.linspace(1e-3, 1, 5)
    if strat == "mcp":
        alphas = np.linspace(1.5, 10, 5)
    elif strat == "scad":
        alphas = np.linspace(2.5, 10, 5)

    best_lambs = []
    best_alphas = []
    for _ in range(rep):
        best = 1e10
        seed = np.random.randint(100)
        rds = np.random.RandomState(seed)
        U = gen_orth_mat(p, s, rds)
        Pi = U[:,:k] @ U[:,:k].T
        es = np.ones(p)
        for i in range(k):
            if i == k - 1:
                es[i] = 10
            else:
                es[i] = 100
        E = np.diag(es)
        Sigma = U @ E @ U.T

        Xs = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        Sigma_h = Xs.T @ Xs / n
        for lamb in lambs:
            for alpha in alphas:
                if strat == "mcp":
                    Pi_h, _ = mm(Sigma_h, lamb, alpha, k, 9, 100, 20)
                elif strat == "scad":
                    Pi_h, _ = mm(Sigma_h, lamb, alpha, k, 9, 100, 20, "scad")
                if np.linalg.norm(Pi - Pi_h) < best:
                    best_lamb = lamb
                    best_alpha = alpha
                    best = np.linalg.norm(Pi - Pi_h)
        best_lambs.append(best_lamb)
        best_alphas.append(best_alpha)
    best_lambs = np.asarray(best_lambs)
    best_alphas = np.asarray(best_alphas)

    # save the result
    result = {"k": k, "s": s, "p": p, "n": n, "best_lambs": best_lambs.tolist(), "best_alphas": best_alphas.tolist(), "strat": strat}
    with open(os.path.join(path, "mm", "grid.log"), "a") as f:
        json.dump(result, f)
        f.write(",\n")

    # np.save(os.path.join(path, "k=%i_s=%i_p=%i_n=%i_best_lambs.npy" % (k, s, p, n)), best_lambs)
    # np.save(os.path.join(path, "k=%i_s=%i_p=%i_n=%i_best_alphas.npy" % (k, s, p, n)), best_alphas)
    print("best: %f" % best)


def grid_Gu(rep, k, s, p, n, path):
    lambs = np.linspace(1e-3, 1, 5)
    alphas = np.linspace(1.5, 10, 5)

    best_lambs = []
    best_alphas = []
    for _ in range(rep):
        best = 1e10
        seed = np.random.randint(100)
        rds = np.random.RandomState(seed)
        U = gen_orth_mat(p, s, rds)
        Pi = U[:,:k] @ U[:,:k].T
        es = np.ones(p)
        for i in range(k):
            if i == k - 1:
                es[i] = 10
            else:
                es[i] = 100
        E = np.diag(es)
        Sigma = U @ E @ U.T

        Xs = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        Sigma_h = Xs.T @ Xs / n
        for lamb in lambs:
            for alpha in alphas:
                Pi_h, _ = Gu(Sigma_h, k, 0.5, 1, lamb, alpha, 2000, p, Pi, strategy="mcp")
                if np.linalg.norm(Pi - Pi_h) < best:
                    best_lamb = lamb
                    best_alpha = alpha
                    best = np.linalg.norm(Pi - Pi_h)
        best_lambs.append(best_lamb)
        best_alphas.append(best_alpha)
    best_lambs = np.asarray(best_lambs)
    best_alphas = np.asarray(best_alphas)

    result = {"k": k, "s": s, "p": p, "n": n, "best_lambs": best_lambs.tolist(), "best_alphas": best_alphas.tolist()}
    with open(os.path.join(path, "Gu", "grid.log"), "a") as f:
        json.dump(result, f)
        f.write(",\n")
    # np.save(os.path.join(path, "k=%i_s=%i_p=%i_n=%i_best_lambs.npy" % (k, s, p, n)), best_lambs)
    # np.save(os.path.join(path, "k=%i_s=%i_p=%i_n=%i_best_alphas.npy" % (k, s, p, n)), best_alphas)
    print("best: %f" % best)

def grid_l1(rep, k, s, p, n, path):
    # special case of Gu
    ord = np.arange(-9, 2)
    lambs = 10. ** ord
    best_lambs = []
    for _ in range(rep):
        best = 1e10
        seed = np.random.randint(100)
        rds = np.random.RandomState(seed)
        U = gen_orth_mat(p, s, rds)
        Pi = U[:,:k] @ U[:,:k].T
        es = np.ones(p)
        for i in range(k):
            if i == k - 1:
                es[i] = 10
            else:
                es[i] = 100
        E = np.diag(es)
        Sigma = U @ E @ U.T

        Xs = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        Sigma_h = Xs.T @ Xs / n
        for lamb in lambs:
            Pi_h, _ = Gu(Sigma_h, k, 0, 1, lamb, 1, 2000, p, Pi, strategy="l1")
            if np.linalg.norm(Pi - Pi_h) < best:
                best_lamb = lamb
                best = np.linalg.norm(Pi - Pi_h)
        best_lambs.append(best_lamb)

    best_lambs = np.asarray(best_lambs)
    result = {"k": k, "s": s, "p": p, "n": n, "best_lambs": best_lambs.tolist()}
    with open(os.path.join(path, "l1", "grid.log"), "a") as f:
        json.dump(result, f)
        f.write(",\n")
    # np.save(os.path.join(path, "k=%i_s=%i_p=%i_n=%i_best_lambs.npy" % (k, s, p, n)), best_lambs)
    print("best: %f" % best)


def error_exp(rep, alg, k, s, p, n, path, best_args, H_SNR=False):
    # list of seeds
    seeds = list(range(rep))

    errors = []
    cup_times = []
    TPRs = []
    FPRs = []

    # create the covariance matrix
    U, supp = gen_orth_mat(p, s, k, 0)
    assert np.linalg.matrix_rank(U) == p
    Pi = U[:,:k] @ U[:,:k].T
    if H_SNR:
        es = 0.1 * np.ones(p)
        for i in range(k):
            es[i] = 100
    else:
        es = np.ones(p)
        for i in range(k):
            if i == k - 1:
                es[i] = 10
            else:
                es[i] = 100

    E = np.diag(es)
    Sigma = U @ E @ U.T

    for seed in seeds:
        # set the random seed
        np.random.seed(seed)

        Xs = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        Sigma_h = Xs.T @ Xs / n

        start_time = time.time()
        if alg == "mm":
            Pi_h, _ = mm(Sigma_h, best_args["scale"], best_args["lamb"], best_args["alpha"], k, best_args["rho"], best_args["outer"], best_args["inner"])
        elif alg == "Gu":
            Pi_h, _ = Gu(Sigma_h, k, best_args["tau"], best_args["rho"], best_args["lamb"], best_args["alpha"], best_args["iter"], p, Pi, strategy="mcp")
        elif alg == "l1":
            Pi_h, _ = Gu(Sigma_h, k, best_args["tau"], best_args["rho"], best_args["lamb"], 1, best_args["iter"], p, Pi, strategy="l1")
        elif alg == "Wang":
            Pi_h, _ = Wang(Sigma_h, k, 1, best_args["tau"], best_args["lamb"], best_args["scale"], p, s, Pi, best_args["pre"], best_args["post"])
        elif alg == "mm-scad":
            Pi_h, _ = mm(Sigma_h, 1, best_args["lamb"], best_args["alpha"], k, 9, 20, 100, "scad")
        elif alg == "oracle":
            Pi_h = oracle(Sigma_h, k, supp)
        elif alg == "PCA":
            Pi_h = PCA(Sigma_h, k)

        end_time = time.time()
        cup_times.append(end_time - start_time)
        error = np.linalg.norm(Pi_h - Pi)

        print("error:", error)

        errors.append(error)

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
        TPRs.append(tp / (tp + fn))
        FPRs.append(fp / (fp + tn))

        print("TPR:", tp / (tp + fn))
        print("FPR:", fp / (fp + tn))



    result = {"rep": rep, "k": k, "s": s, "p": p, "n": n, "errors": errors, "TPRs": TPRs, "FPRs": FPRs, "H_SNR": H_SNR, "seeds": seeds, "cup_times": cup_times, "time": str(datetime.datetime.now())}
    with open(os.path.join(path, alg, "error.log"), "a") as f:
        json.dump(result, f)
        f.write(",\n")


def crit_exp():
    # TODO
    # random seed
    seed = np.random.randint(100)
    rds = np.random.RandomState(seed)
    np.random.seed(seed)

    # step1: data preparation
    # setting: p=128, s=5, k=1

    # the scale of the problem
    p = 128
    # the number of samples
    n = 80
    # the sparsity
    s = 5
    # the dimension of eigenspace
    k = 1


    U = gen_orth_mat(p, s, rds, arti=True)

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


    np.set_printoptions(precision=2, suppress=True)

    # step3: solving by ADMM
    # best lambda=0.1, tau=0.5, rho=1e-5, alpha=5

    _, _, crit_hist = mm(Sigma_h, 0.35025, 1.5, k, 200, 100, 200, crit=True)
        


def rate_verify(path, rep, k, p, n, num_s, best_args, arti=False):
    """
    prove the rate is tight
    """
    seeds = list(range(rep))
    cord_x = np.array([i * np.sqrt(1 / n) for i in range(k, k + num_s)])
    cord_ys = []
    for seed in seeds:
        cord_y = []
        for s in range(5 + k, 5 + k + num_s):
            rds = np.random.RandomState(seed)
            np.random.seed(seed)
            U = gen_orth_mat(p, s, rds, arti)
            Pi = U[:,:k] @ U[:,:k].T
            es = np.ones(p)
            for i in range(k):
                if i == k - 1:
                    es[i] = 10
                else:
                    es[i] = 100
            E = np.diag(es)
            Sigma = U @ E @ U.T

            Xs = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
            Sigma_h = Xs.T @ Xs / n
            Pi_h, _ = mm(Sigma_h, best_args["lamb"], best_args["alpha"], k, 9, 50, 5)
            cord_y.append(np.linalg.norm(Pi_h - Pi))
        
        cord_ys.append(cord_y)
    avg_cord_y = np.mean(np.array(cord_ys), axis=0)
    result = {"rep": rep, "k": k,  "p": p, "n": n, "num_s": num_s, "arti": arti, "cord_x": cord_x.tolist(), "avg_cord_y": avg_cord_y.tolist(), "time": str(datetime.datetime.now())}
    with open(os.path.join(path, "verification", "record.log"), "a") as f:
        json.dump(result, f)
        f.write(",\n")

def crit_conv():
    # random seed
    seed = np.random.randint(100)
    np.random.seed(seed)

    # step1: data preparation
    # setting: p=128, s=5, k=1

    # the scale of the problem
    p = 100
    # the number of samples
    n = 120
    # the sparsity
    s = 10
    # the dimension of eigenspace
    k = 5

    U, _ = gen_orth_mat(p, s, k, 0)
    assert np.linalg.matrix_rank(U) == p

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

    np.set_printoptions(precision=2, suppress=True)

    # step3: solving by ADMM
    # best lambda=0.1, tau=0.5, rho=1e-5, beta=5

    _, _, crit_hist = mm(Sigma_h, 1, 0.5005, 3.625, k, 20, 1, 1000, crit=True)

    np.save("record/criterion/crit_hist.npy", crit_hist)

def contraction():
    # random seed
    np.random.seed(0)

    # step1: data preparation
    # setting: p=128, s=5, k=1

    # the scale of the problem
    p = 100
    # the number of samples
    n = 120
    # the sparsity
    s = 10
    # the dimension of eigenspace
    k = 5

    U, _ = gen_orth_mat(p, s, k, 0)
    assert np.linalg.matrix_rank(U) == p
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
    
    _, hist = mm(Sigma_h, 1, 0.5005, 3.625, k, 20, 20, 50)
    
    est_err = []
    for Pi_h in hist:
        # restore the projector
        evalue, evector = np.linalg.eigh(Pi_h)
        
        # # judge whether oracle
        # print("evalues:", evalue)

        # restore the projector
        idx = np.argsort(np.real(-evalue))
        evector = evector[:, idx]
        proj = evector[:, :k] @ evector[:, :k].T
        est_err.append(np.linalg.norm(Pi-proj))

    np.save("record/contraction/err_hist.npy", est_err)

if __name__ == "__main__":
    # grid search
    # grid_mm(1, 5, 10, 128, 80, "results/grid_mm")
    # grid_Gu(1, 5, 10, 128, 80, "results/grid_Gu")
    # grid_l1(1, 5, 10, 128, 80, "results/grid_l1")

    # grid search with different setting
    # grid_mm(1, 1, 5, 128, 80, "results/grid_mm")
    # grid_Gu(1, 1, 5, 128, 80, "results/grid_Gu")
    # grid_l1(1, 1, 5, 128, 80, "results/grid_l1")

    # grid_mm(1, 10, 20, 256, 160, "results")
    # grid_Gu(1, 10, 20, 256, 80, "results/grid_Gu")
    # grid_l1(1, 10, 20, 256, 160, "results")

    # grid_mm(1, 50, 100, 1000, 625, "results")
    # grid_Gu(1, 50, 100, 1000, 80, "results")
    # grid_l1(1, 50, 100, 1000, 625, "results")
    
    # error_exp(20, "mm", 1, 5, 128, 80, "results", {"alpha": 1.5, "lamb": 0.35025}, True)
    # error_exp(20, "l1", 1, 5, 128, 80, "results", {"lamb": 0.1}, True)
    # error_exp(20, "Gu", 1, 5, 128, 80, "results", {"alpha": 3.625, "lamb": 0.5005}, True)
    # error_exp(20, "mm", 1, 7, 128, 80, "results", {"alpha": 10.0, "lamb": 0.25075}, True)
    # error_exp(20, "mm", 1, 9, 128, 80, "results", {"alpha": 10.0, "lamb": 0.25075}, True)
    # error_exp(20, "l1", 1, 7, 128, 80, "results", {"lamb": 0.1}, True)
    # error_exp(20, "l1", 1, 9, 128, 80, "results", {"lamb": 0.1}, True)
    # error_exp(20, "Gu", 1, 7, 128, 80, "results", {"alpha": 3.625, "lamb": 0.5005}, True)
    # error_exp(20, "Gu", 1, 9, 128, 80, "results", {"alpha": 3.625, "lamb": 0.5005}, True)
    # error_exp(20, "mm", 5, 10, 128, 80, "results", {"alpha": 3.625, "lamb": 0.75025})





    # # run in parallel
    # import concurrent.futures

    # # Define your function calls as a list of tuples
    # function_calls = [
    #     (20, "mm", 5, 10, 200, 80, "results", {"alpha": 1.5, "lamb": 0.35025}, True),
    #     (20, "Gu", 5, 10, 200, 80, "results", {"tau": 1, "alpha": 3.625, "lamb": 0.5005}, True),
    #     (20, "l1", 5, 10, 200, 80, "results", {"lamb": 0.1}, True),
    #     (20, "Wang", 5, 10, 200, 80, "results", {"lamb": 0.1}, True),
    #     # (20, "mm", 5, 10, 200, 80, "results", {"alpha": 1.5, "lamb": 0.35025}, False),
    #     # (20, "Gu", 5, 10, 200, 80, "results", {"tau": 1, "alpha": 3.625, "lamb": 0.5005}, False),
    #     # (20, "l1", 5, 10, 200, 80, "results", {"lamb": 0.1}, False),
    #     # (20, "Wang", 5, 10, 200, 80, "results", {"lamb": 0.1}, False),
    # ]

    # # Define a wrapper function that handles the H_SNR parameter
    # def run_error_exp(*args, H_SNR):
    #     if H_SNR:
    #         error_exp(*args, H_SNR=True)
    #     else:
    #         error_exp(*args)

    # # Run the function calls in parallel using ThreadPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(run_error_exp, *args[:-1], H_SNR=args[-1]) for args in function_calls]

    #     # Optionally, you can wait for all tasks to complete and get the results (if any)
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             result = future.result()  # If error_exp returns something, otherwise just pass
    #         except Exception as e:
    #             print(f"An error occurred: {e}")

    ###### experiment setting 1
    # error_exp(20, "Gu", 5, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(20, "mm", 5, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False) # best one
    # error_exp(20, "l1", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.5, "pre": 80,  "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "PCA", 5, 10, 100, 60, "results", {}, H_SNR=False)

    # error_exp(20, "Gu", 5, 10, 100, 80, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(20, "mm", 5, 10, 100, 80, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False) # best one
    # error_exp(20, "l1", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 1, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.5, "pre": 80,  "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "PCA", 5, 10, 100, 80, "results", {}, H_SNR=False)

    # error_exp(20, "Gu", 5, 10, 100, 120, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(20, "mm", 5, 10, 100, 120, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False) # best one
    # error_exp(20, "l1", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 1, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.5, "pre": 80,  "post": 20}, H_SNR=False)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(20, "PCA", 5, 10, 100, 120, "results", {}, H_SNR=False)
   

    ###### experiment setting 2
    # Gu
    # error_exp(5, "Gu", 1, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 2, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 3, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 4, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 5, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 6, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 7, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 8, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    
    # MM
    # error_exp(5, "mm", 1, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 2, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 3, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 4, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 5, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 6, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 7, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 8, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    
    # L1
    # error_exp(5, "l1", 1, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 2, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 3, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 4, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 6, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 7, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 8, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    
    # Wang
    # error_exp(1, "Wang", 1, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(1, "Wang", 2, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(1, "Wang", 3, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(1, "Wang", 4, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(1, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(1, "Wang", 6, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(1, "Wang", 7, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    # error_exp(1, "Wang", 8, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=False)
    
    ###### experiment setting 3

    # error_exp(20, "Gu", 5, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=True)
    # error_exp(20, "mm", 5, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=True) # best one
    # error_exp(20, "l1", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.5, "pre": 80,  "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005, "scale": 2, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "PCA", 5, 10, 100, 60, "results", {}, H_SNR=True)

    # error_exp(20, "Gu", 5, 10, 100, 80, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=True)
    # error_exp(20, "mm", 5, 10, 100, 80, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=True) # best one
    # error_exp(20, "l1", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 1, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.5, "pre": 80,  "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 80, "results", {"tau": 10, "lamb": 0.5005, "scale": 2, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "PCA", 5, 10, 100, 80, "results", {}, H_SNR=True)

    # error_exp(20, "Gu", 5, 10, 100, 120, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=True)
    # error_exp(20, "mm", 5, 10, 100, 120, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=True) # best one
    # error_exp(20, "l1", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 1, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.2, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 1.5, "pre": 80,  "post": 20}, H_SNR=True)
    # error_exp(20, "Wang", 5, 10, 100, 120, "results", {"tau": 10, "lamb": 0.5005, "scale": 2, "pre": 80, "post": 20}, H_SNR=True)
    # error_exp(20, "PCA", 5, 10, 100, 120, "results", {}, H_SNR=True)


    ###### experiment setting 4
    # crit_conv()


    ###### experiment setting 5 
    # Gu
    # error_exp(5, "Gu", 1, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 2, 10, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 3, 20, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 4, 20, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 5, 20, 100, 60, "results", {"tau": 10,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 6, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 7, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 8, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 9, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 10, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "Gu", 11, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)

    # MM
    # error_exp(5, "mm", 1, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 2, 10, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 3, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 4, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 5, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 6, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 7, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 8, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 9, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 10, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    # error_exp(5, "mm", 11, 20, 100, 60, "results", {"scale": 1, "rho": 20, "alpha": 3.625, "lamb": 0.5005, "outer": 5, "inner": 20}, H_SNR=False)
    
    # L1
    # error_exp(5, "l1", 1, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 2, 10, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 3, 20, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 4, 20, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 5, 20, 100, 60, "results", {"tau": 10, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 6, 20, 100, 60, "results", {"tau": 20, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 7, 20, 100, 60, "results", {"tau": 20, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 8, 20, 100, 60, "results", {"tau": 20, "lamb": 0.5005,  "rho": 9, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 9, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 10, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)
    # error_exp(5, "l1", 11, 20, 100, 60, "results", {"tau": 20,  "rho": 9, "alpha": 3.625, "lamb": 0.5005, "iter": 100}, H_SNR=False)

    ###### experiment setting 6
    # contraction()
    crit_conv()
