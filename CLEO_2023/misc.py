import clipboard_and_style_sheet
import matplotlib.pyplot as plt
import numpy as np

# %% -----
fig, ax = plt.subplots(1, 1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([10**i for i in range(-2, 7)])
ax.set_yticks([10**i for i in range(1, 5)])
ax.set_xlabel("Spectral Acquisition Rate (Hz)")
ax.set_ylabel("Optical Bandwidth ($\\mathrm{cm^{-1}}$)")
ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()
