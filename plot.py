#### contraction
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import *
from algs import *


plt.rcParams['text.usetex'] = True

# Set the style of the plot
sns.set_theme(style="whitegrid")

# Create the plot
plt.figure(figsize=(6, 4))

# plt.plot(crit_hist[0], color='b', marker='o', linestyle='-', markevery=5, label=r'$k=1$')

# The template: plt.plot(crit_hist[0], linewidth=5, color='g', marker='^', linestyle='-', markevery=5)

err_hist = np.load("record/contraction/err_hist.npy")
y = err_hist[1: ]
x = np.arange(1, 21)

# linewidth
# linewidth = 6 # default
linewidth = 2

plt.plot(x, y, linewidth=linewidth, color="#00796B")

# plt.plot(x, y, marker='^', linestyle='None', markersize=12, color='#FFE0B2', 
#          markeredgecolor='#9E9E9E', markevery=1, markeredgewidth=2)

# shape of the marker
# m = 'p'
m = ''
# m = '^'
# plt.plot(x, y, marker=m, linestyle='None', markersize=12, color='#f5af19', 
#          markeredgecolor='None', markevery=1, markeredgewidth=3, label="Every 5th Point")
plt.plot(x[0], y[0], marker=m, markersize=15, linestyle='None', color='#DBF227', 
         markeredgecolor='#D6D58E', markeredgewidth=4, zorder=2)
plt.plot(x[-1], y[-1], marker=m, markersize=15, linestyle='None', color='#DBF227', 
         markeredgecolor='#005C53', markeredgewidth=4, zorder=2)

# plt.plot(x, crit_hist[0], linewidth=6, color='#11998e', marker='^', linestyle='-', markevery=5, markersize=12, markerfacecolor='#38ef7d', markeredgecolor='#0575E6')

grid = True

plt.grid(grid)

plt.gca().spines['top'].set_zorder(0)
plt.gca().spines['bottom'].set_zorder(0)
plt.gca().spines['left'].set_zorder(0)
plt.gca().spines['right'].set_zorder(0)

plt.gca().spines['left'].set_linewidth(2.5)
plt.gca().spines['bottom'].set_linewidth(2.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')

# TODO: set the origin point.
# plt.gca().spines['left'].set_position(('data', 0))
# plt.gca().spines['bottom'].set_position(('data', 0))



# Add titles and labels
plt.xlabel(r'$k$', fontsize=28)
plt.ylabel(r'$\left\Vert\cdot\right\Vert_{\mathsf{F}}$', fontsize=25, labelpad=-2)

# Add a legend
# plt.legend()

# Customize the ticks
xticks = np.arange(1, 22, 2)
plt.xlim(0.4, 20.8)
plt.gca().set_xticks(xticks)
plt.yticks(np.arange(0.182, 0.235, 0.01))
plt.ylim(0.180, 0.215)


plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

start_x, start_y = x[0], y[0]
#### TODO: annotate
# plt.annotate(f'({int(start_x)}, {start_y:.3f})', xy=(start_x, start_y), xytext=(start_x + 0.8, start_y), fontsize=22)

# second_x, second_y = x[1], y[1]
# plt.annotate(f'({int(second_x)}, {second_y:.3f})', xy=(second_x, second_y), xytext=(second_x + 0.2, second_y + 0.002), fontsize=22)

end_x, end_y = x[-1], y[-1]
#### TODO: annotate
# plt.annotate(f'({int(end_x)}, {end_y:.3f})', xy=(end_x, end_y), xytext=(end_x-5, end_y + 0.003), fontsize=22)

# TODO: 水平箭头
plt.annotate('', xy=(21, 0.180), xytext=(0.4, 0.180),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5, mutation_scale=30))

# TODO: 垂直箭头
plt.annotate('', xy=(0.4, 0.215), xytext=(0.4, 0.180),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5, mutation_scale=30))

plt.savefig("record/contraction/result.eps", format='eps')

print(err_hist[0], err_hist[1])
