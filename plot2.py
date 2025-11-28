#### comparison (6-11)
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True

def plot_pic_mean_median(data1,data2,data3,title,xlabel,ylabel='',ylim = None):
    # Sample data points, which you should replace with your actual data
    ee = np.array([6,7,8,9,10,11])

    # Creating the plot
    plt.figure(figsize=(6, 4))
    plt.plot(ee, data1, marker='s', linestyle='--', color = "#7B1FA2", markeredgecolor='#5989a5', markerfacecolor='none', markeredgewidth=2, label="CVX. ADMM", linewidth=2, zorder=3)
    plt.plot(ee, data2, marker='o', linestyle='-.', color = "#103778", markeredgecolor='#512DA8', markerfacecolor='none', markeredgewidth=2, label="NCVX. ADMM", linewidth=2, zorder=3)
    plt.plot(ee, data3, marker='*', linestyle='-', color = "#009688", markeredgecolor='#00796B', markerfacecolor='none', markeredgewidth=2, markersize=10, label="MM", linewidth=2, zorder=3)
    
    # TODO plt.grid(axis='y')
    plt.grid(True)


    # Adding title and labels
    #plt.title(title)
    plt.xlabel(xlabel, labelpad=-5, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)

    # Adding a legend
    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(5.8, 11.2)
    plt.xticks(np.arange(6.0, 11.1, 1))
    # Show grid
    # plt.grid(False)
    if ylim:
        plt.ylim(ylim[0], ylim[1])
        step = (ylim[1] - ylim[0]) / 5
        plt.yticks(np.arange(ylim[0], ylim[1] + 0.001, step))
    # Display the plot
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.tight_layout()
    plt.savefig("record/comparison/result_s=20.eps", format='eps')

if __name__ == '__main__':
    data1 = np.array([0.4452, 0.4410, 0.4968, 0.4836, 0.5349, 0.5608])
    data2 = np.array([0.4376, 0.4331, 0.4766, 0.4865, 0.5356, 0.5505])
    data3 = np.array([0.3856, 0.4156, 0.4410, 0.4368, 0.5087, 0.5368])
    plot_pic_mean_median(data1,data2,data3, "", r'$r$', r"$\left\Vert\cdot\right\Vert _{\mathsf{F}}$", [0.3,0.6])
