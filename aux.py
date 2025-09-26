import numpy as np

def mcp(Pi, lamb, alpha):
    # alpha is required to be larger than 1
    if alpha <= 1:
        raise Exception("mcp: parameter 'alpha' should be larger than 1")

    t_0 = alpha * lamb
    out = np.multiply(lamb - Pi / alpha, np.where(Pi<=t_0, 1, 0))
    return out

def scad(Pi, lamb, alpha=3.7):
    # alpha is required to be larger than 2
    if alpha <= 2:
        raise Exception("scad: parameter 'alpha' should be larger than 2")
    
    # def fun(x, a, b):
    #     if x < a:
    #         return a
    #     elif a <= x < a * b:
    #         return (a * b - x) / (b - 1)
    #     else:
    #         return 0

    # v_fun = np.vectorize(fun)
    # return v_fun(Pi, lamb, alpha)

    res = np.zeros(Pi.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            x = Pi[i][j]
            if x < lamb:
                res[i][j] = lamb
            elif lamb <= x < lamb * alpha:
                res[i][j] = (alpha * lamb - x) / (alpha - 1)
            else:
                res[i][j] = 0

    return res
    