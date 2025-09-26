#### comparison (3-8)
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True

def plot_pic_mean_median(data1,data2,data3,title,xlabel,ylabel,ylim = None):
    # Sample data points, which you should replace with your actual data
    ee = np.array([3,4,5,6,7,8])

    # Creating the plot
    plt.figure(figsize=(6, 4))
    plt.plot(ee, data1, marker='s', linestyle='--', color = "#7B1FA2", markeredgecolor='#5989a5', markerfacecolor='none', markeredgewidth=2, label="CVX. ADMM", linewidth=2, zorder=3)
    plt.plot(ee, data2, marker='o', linestyle='-.', color = "#103778", markeredgecolor='#512DA8', markerfacecolor='none', markeredgewidth=2, label="NCVX. ADMM", linewidth=2, zorder=3)
    plt.plot(ee, data3, marker='*', linestyle='-', color = "#009688", markeredgecolor='#00796B', markerfacecolor='none', markeredgewidth=2, markersize=10, label="MM (prop.)", linewidth=2, zorder=3)
    # TODO plt.grid(axis='y')
    plt.grid(True)

    # Adding title and labels
    #plt.title(title)
    plt.xlabel(xlabel, labelpad=-5, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)

    # Adding a legend
    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(2.8, 8.2)
    plt.xticks(np.arange(3, 9, 1))
    # Show grid
    # plt.grid(False)
    if ylim:
        plt.ylim(ylim[0], ylim[1])
        step = (ylim[1] - ylim[0]) / 4
        plt.yticks(np.arange(ylim[0], ylim[1] + 0.001, step))
    # Display the plot
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    plt.savefig("record/comparison/result.eps", format='eps')

if __name__ == '__main__':
    data1 = np.array([0.2755,0.2669,0.2939,0.3307,0.3492,0.3467])
    data2 = np.array([0.2486,0.2597,0.2838,0.3372,0.3474,0.3487])
    data3 = np.array([0.2430,0.2561,0.2788,0.3147,0.3294,0.3411])
    plot_pic_mean_median(data1,data2,data3," ", r'$r$', r"$\left\Vert\cdot\right\Vert _{\mathsf{F}}$",[0.2,0.4])
