#### stopping criterion * iterations
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

crit_hist = np.load("record/criterion/crit_hist.npy")

y = crit_hist[0]
x = np.arange(1, 1001)

# plot the product of criterion and iterations
y = y * x

# shape of the marker
# m = 'p'
m = ''
# m = '^'
# linewidth
# linewidth = 6 # default
linewidth = 2
plt.plot(x, y, linewidth=linewidth, color="#4286f4")

# plt.plot(x, y, marker='^', linestyle='None', markersize=12, color='#f5af19', 
#          markeredgecolor='#2c3e50', markevery=5, markeredgewidth=3, label="Every 5th Point")

plt.plot(x[0], y[0], marker=m, markersize=15, linestyle='None', color='#f5af19', 
         markeredgecolor='#F37335', markeredgewidth=3, label="Start Point")

plt.plot(x[-1], y[-1], marker=m, markersize=15, linestyle='None', color='#f5af19', 
         markeredgecolor='#4e54c8', markeredgewidth=3, label="End Point")


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

# fix to the origin
plt.gca().spines['left'].set_position(('data', 0))
plt.gca().spines['bottom'].set_position(('data', 0))



# Add titles and labels
plt.xlabel(r'$t$', fontsize=28)
plt.ylabel(r'$\omega_{t}\times t$', fontsize=28, labelpad=0)

# Add a legend
# plt.legend()

# Customize the ticks
xticks = np.arange(0, 1010, 100)
xticks = xticks[1:]
plt.xlim(0, 1008)
plt.ylim(0, 20)

plt.gca().set_xticks(xticks)
plt.yticks(np.arange(0, 50, 6))

plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

start_x, start_y = x[0], y[0]
# TODO annotate
# plt.annotate(f'({int(start_x)}, {start_y:.1f})', xy=(start_x, start_y), xytext=(start_x + 2, start_y), fontsize=22)

end_x, end_y = x[-1], y[-1]
# TODO annotate
# plt.annotate(f'({int(end_x)}, {end_y:.1f})', xy=(end_x, end_y), xytext=(end_x - 15, end_y + 1), fontsize=22)
plt.savefig("record/criterion/result(prod).eps", format='eps')
