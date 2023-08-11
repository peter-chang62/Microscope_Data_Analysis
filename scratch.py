import numpy as np
import matplotlib.pyplot as plt
import clipboard as cr


# %% -----
fig, ax = plt.subplots(1, 1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([10**i for i in range(-2, 7)])
ax.set_yticks([10**i for i in range(1, 5)])
ax.set_xlabel("Spectral Acquisition Rate (Hz)")
ax.set_ylabel("Optical Bandwidth ($\\mathrm{cm^{-1}}$)")
ax.spines[["right", "top"]].set_visible(False)

ax.plot([10**-2, 10**7][::-1], [1, 10**5], "--")
ax.set_xlim(0.01, 1000000.0)
ax.set_ylim(0.8912509381337456, 10000.0)
fig.tight_layout()
