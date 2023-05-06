import clipboard_and_style_sheet as cr
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
ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()

# %% -----
# cr.style_sheet()
fig, ax = plt.subplots(1, 1)
dfrep = np.linspace(5000, 100e3, 5000)
bandwidth = lambda fr, dfr: fr**2 / (2 * dfr)
ax.plot(dfrep * 1e-3, bandwidth(100e6, dfrep) * 1e-12, linewidth=3, label="100 MHz")
ax.plot(dfrep * 1e-3, bandwidth(1000e6, dfrep) * 1e-12, linewidth=3, label="1 GHz")
ax.plot(dfrep * 1e-3, bandwidth(200e6, dfrep) * 1e-12, linewidth=3, label=" 200MHz")
ax.legend(loc="best")
ax.set_xlabel("acquisition speed kHz")
ax.set_ylabel("$\\mathrm{\\Delta \\nu (THz)}$")
fig.tight_layout()
