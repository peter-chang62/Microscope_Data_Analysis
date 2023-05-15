# %% ----- package imports
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt
import numpy as np


def bandwidth(fr, dfr):
    return fr**2 / (2 * dfr)


figsize = np.array([3.99, 2.83])

# %% ----- background axes for intro slide
fig, ax = plt.subplots(1, 1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([10**i for i in range(-2, 7)])
ax.set_yticks([10**i for i in range(1, 5)])
ax.set_xlabel("Spectral Acquisition Rate (Hz)")
ax.set_ylabel("Optical Bandwidth ($\\mathrm{cm^{-1}}$)")
ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()

# %% ----- 100 MHz & 200 MHz
fig, ax = plt.subplots(1, 1, figsize=figsize)
dfrep = np.linspace(50, 1000, 5000)
ax.semilogy(
    dfrep,
    bandwidth(100e6, dfrep) * 1e-12,
    linewidth=3,
    label="100 MHz",
    color="C0",
)
ax.semilogy(
    dfrep,
    bandwidth(200e6, dfrep) * 1e-12,
    linewidth=3,
    label=" 200MHz",
    color="C2",
)
ax.axvline(200, color="k", linestyle="--", linewidth=2)
ax.legend(loc="best")
ax.set_xlabel("acquisition speed for $\\mathrm{\\Delta \\nu}$ (Hz)")
ax.set_ylabel("$\\mathrm{\\Delta \\nu (THz)}$")
fig.tight_layout()

# %% ----- 100 MHz, 200 MHz, & 1 GHz
fig, ax = plt.subplots(1, 1, figsize=figsize)
dfrep = np.linspace(50e2, 1000e2, 5000)
scale = 1e3
ax.semilogy(
    dfrep / scale,
    bandwidth(100e6, dfrep) * 1e-12,
    linewidth=3,
    label="100 MHz",
    color="C0",
)
ax.semilogy(
    dfrep / scale,
    bandwidth(200e6, dfrep) * 1e-12,
    linewidth=3,
    label=" 200MHz",
    color="C2",
)
ax.semilogy(
    dfrep / scale,
    bandwidth(1e9, dfrep) * 1e-12,
    linewidth=3,
    label="1 GHz",
    color="C1",
)
ax.axvline(20, color="k", linestyle="--", linewidth=2)
ax.legend(loc="best")
ax.set_xlabel("acquisition speed for $\\mathrm{\\Delta \\nu}$ (kHz)")
ax.set_ylabel("$\\mathrm{\\Delta \\nu (THz)}$")
fig.tight_layout()

# %% ----- 100 MHz, 200 mHz, 1 GHz & 10 GHz
fig, ax = plt.subplots(1, 1, figsize=figsize)
dfrep = np.linspace(50e4, 1000e4, 5000)
scale = 1e6
ax.semilogy(
    dfrep / scale,
    bandwidth(100e6, dfrep) * 1e-12,
    linewidth=3,
    label="100 MHz",
    color="C0",
)
ax.semilogy(
    dfrep / scale,
    bandwidth(200e6, dfrep) * 1e-12,
    linewidth=3,
    label=" 200MHz",
    color="C2",
)
ax.semilogy(
    dfrep / scale,
    bandwidth(1e9, dfrep) * 1e-12,
    linewidth=3,
    label="1 GHz",
    color="C1",
)
ax.semilogy(
    dfrep / scale,
    bandwidth(10e9, dfrep) * 1e-12,
    linewidth=3,
    label="10 GHz",
    color="C3",
    linestyle="--",
)
ax.axvline(2, color="k", linestyle="--", linewidth=2)
ax.legend(loc="best")
ax.set_xlabel("acquisition speed for $\\mathrm{\\Delta \\nu}$ (MHz)")
ax.set_ylabel("$\\mathrm{\\Delta \\nu (THz)}$")
fig.tight_layout()
