### plot difference between spiked and non-spiked covariance matrix

import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import seaborn as sns


def gen_cluster_orth_mat(p, s, k, seed=0):
    """
    Randomly generate a p * p orthogonormal matrix with the first k columns has sparsity s
    """
    rds = np.random.RandomState(seed=seed)
    U1 = ortho_group.rvs(s, random_state=rds)[:,:k]
    insert_indices = np.arange(s, p)
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
    # check the feasibility
    assert np.linalg.matrix_rank(U3) == p
    supp = np.setdiff1d(np.arange(p), insert_indices)
    return U3, supp


p = 10
s = 4
k = 2
val = 2
U, _ = gen_cluster_orth_mat(p, s, k, 0)

spike = False
if spike:
    es = np.ones(p) * val
else:
    es = np.random.uniform(low=1e-9, high=2 * val-1e-9, size=p)
    es = np.sort(es)[::-1]
for i in range(k):
    if i == k - 1:
        es[i] = 5
    else:
        es[i] = 10

E = np.diag(es)
Sigma = U @ E @ U.T
signal_part = U[:,:k] @ E[:k, :k] @ U[:,:k].T
nonsignal_part = U[:,k:] @ E[k:, k:] @ U[:,k:].T
def plot_custom_heatmap(data_matrix, save_pth):
    """
    根据输入矩阵绘制热力图，样式：红蓝双色，无标签刻度，有颜色条。
    """
    # 设置绘图风格
    sns.set_theme(style="white")

    # 调整图表大小
    plt.figure(figsize=(10, 10))

    # 使用 seaborn.heatmap 绘制热力图
    # cmap='coolwarm': 使用红蓝双色调色板
    # center=0: 确保白色中心点落在 0 值上，用于处理正负值对比
    # annot=False: 不显示单元格中的数值
    # xticklabels=False, yticklabels=False: 不显示X轴和Y轴的刻度标签
    # cbar=True: 显示颜色条

    print("Data matrix min:", data_matrix.min())
    print("Data matrix max:", data_matrix.max())

    ax = sns.heatmap(
        data_matrix,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cmap='coolwarm', 
        linewidths=0, # 无单元格边框
        vmin=data_matrix.min(), # 确保颜色条从数据的最小值开始
        vmax=data_matrix.max(), # 确保颜色条到数据的最大值结束
        square=True,
        cbar_kws={'shrink': 0.81, 'pad': 0.04}
    )

    # 调整颜色条的标签大小，使其更美观
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.savefig(save_pth, format='eps')

# 2. 调用函数绘图
if spike:
    cov_save_pth = "record/heatmap/spike_cov_heatmap.eps"
    plot_custom_heatmap(Sigma, cov_save_pth)
    signal_save_pth = "record/heatmap/spike_signal_heatmap.eps"
    plot_custom_heatmap(signal_part, signal_save_pth)
    nonsignal_save_pth = "record/heatmap/spike_nonsignal_heatmap.eps"
    plot_custom_heatmap(nonsignal_part, nonsignal_save_pth)
else:
    cov_save_pth = f"record/heatmap/non-spike_cov_heatmap_p={p}_s={s}_k={k}.eps"
    plot_custom_heatmap(Sigma, cov_save_pth)
    signal_save_pth = f"record/heatmap/spike_signal_heatmap_p={p}_s={s}_k={k}.eps"
    plot_custom_heatmap(signal_part, signal_save_pth)
    nonsignal_save_pth = f"record/heatmap/spike_nonsignal_heatmap_p={p}_s={s}_k={k}.eps"
    plot_custom_heatmap(nonsignal_part, nonsignal_save_pth)
