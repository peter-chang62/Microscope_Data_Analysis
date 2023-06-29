import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(-10, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.fill_between(x, y1, alpha=0.4, step="pre")
plt.fill_between(x, y2, alpha=0.4, step="pre")

plt.semilogy(x, y1, drawstyle="steps")
plt.semilogy(x, y2, drawstyle="steps")

plt.show()
