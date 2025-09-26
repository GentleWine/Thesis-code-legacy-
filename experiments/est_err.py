import json
import time
import numpy as np
import sys
import os
from tqdm import tqdm
import datetime
sys.path.append('../datasets/synthetic')
sys.path.append('../methods')
sys.path.append("..")
from data_gen import config, get_info, origin
from CDRSPCA import cdrspca
from GPM import gpm
from ITSPCA import itspca
from StreamPCA import stream
from utils import NumpyArrayEncoder

def error_exp(rep, alg, src, k, s, p, n, path, best_args):
    config(p, n, s, k)
    # list of seeds
    seeds = list(range(rep))
    errors = []
    TPRs = []
    FPRs = []
    t0 = time.time()
    for seed in tqdm(seeds):
        # data preparation
        if src == "origin":
            X, S, Sigma, Pi, U = origin(seed)
        else:
            raise NotImplementedError(src)
        
        if alg == "cdrspca":
            U_h = cdrspca(S, s, k, p, T = best_args["T"], init_strat=best_args["init_strat"])
            Pi_h = U_h @ U_h.T
        else:
            raise NotImplementedError(alg)

        errors.append(np.linalg.norm(Pi_h - Pi))
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(p):
            for j in range(p):
                if i < s and j < s:
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

    
    setting = {"rep": rep, "k": k, "s": s, "p": p, "n": n, "src": src}
    
    result = {"seeds": seeds, "errors": errors, "TPRs": TPRs, "FPRs": FPRs}
    
    result_dict = {
        'setting': setting,
        'params': best_args,
        'result': result,
        'time': str(datetime.datetime.now())
    }


    folder = os.path.join(path, "error", alg)
    os.makedirs(folder, exist_ok=True)
    exp_hash = str(abs(json.dumps(setting).__hash__()))
    exp_result_file = os.path.join(folder, '%s.json' % exp_hash)
    with open(exp_result_file, 'w') as f:
        json.dump(result_dict, f, indent=4, cls=NumpyArrayEncoder)
    
    print('Dumped results to %s' % exp_result_file)
    print('Duration:', (time.time() - t0) / 60)




if __name__ == "__main__":
    # conduct multiple experiments
    best_args = {}
    best_args["T"] = 100
    best_args["init_strat"] = "random"

    error_exp(5, "cdrspca", "origin", 5, 10, 128, 80, "../record", best_args)

