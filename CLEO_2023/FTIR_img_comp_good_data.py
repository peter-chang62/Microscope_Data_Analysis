# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import clipboard as cr
import os


# %% --------------------------------------------------------------------------
folders = [i.name for i in os.scandir("I3_good_data/")]
folders.remove(".DS_Store")

# %% ----------- plot all of them ---------------------------------------------
for idx in range(0, 6):
    x_i = np.load(f"I3_good_data/{folders[idx]}/img_integrated_absorbance.npy")
    x_bp = np.load(f"I3_good_data/{folders[idx]}/img_big_CH_peak.npy")
    x_sp = np.load(f"I3_good_data/{folders[idx]}/img_small_CH_peak.npy")

    hist, bins = np.histogram(x_bp.flatten(), bins=200)
    hist = hist[1:]  # exclude 0
    bins = bins[1:-1]

    threshold = 0.05
    idx_max = hist.argmax()
    idx_left = abs(hist - hist.max() * threshold)[:idx_max].argmin()
    idx_right = abs(hist - hist.max() * threshold)[idx_max:].argmin() + idx_max
    vmin = bins[idx_left]
    vmax = bins[idx_right]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(
        x_bp[:, ::-1],
        cmap="CMRmap_r_t",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.suptitle(f"data #{idx}, size: {x_bp.shape[0]}x{x_bp.shape[1]}")
    fig.tight_layout()

# %% ----------- plot just one ------------------------------------------------
idx = 0
# dx = 0.7
dx = 3.7
x_i = np.load(f"I3_good_data/{folders[idx]}/img_integrated_absorbance.npy")
x_bp = np.load(f"I3_good_data/{folders[idx]}/img_big_CH_peak.npy")
x_sp = np.load(f"I3_good_data/{folders[idx]}/img_small_CH_peak.npy")

hist, bins = np.histogram(x_bp.flatten(), bins=200)
hist = hist[1:]  # exclude 0
bins = bins[1:-1]

threshold = 0.05
idx_max = hist.argmax()
idx_left = abs(hist - hist.max() * threshold)[:idx_max].argmin()
idx_right = abs(hist - hist.max() * threshold)[idx_max:].argmin() + idx_max
vmin = bins[idx_left]
vmax = bins[idx_right]

fig, ax = plt.subplots(1, 1)
x = np.arange(x_bp.shape[1]) * dx
y = np.arange(x_bp.shape[0]) * dx
ax.pcolormesh(x, y, x_bp[::-1, ::-1], cmap="CMRmap_r_t", vmin=vmin, vmax=vmax)
ax.set_aspect("equal")
fig.suptitle(f"data #{idx}, size: {x_bp.shape[0]}x{x_bp.shape[1]}")

scalebar = AnchoredSizeBar(
    ax.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper left",
    frameon=False,
    color="lightgreen",
    size_vertical=5,
)
ax.add_artist(scalebar)
ax.axis(False)
fig.tight_layout()
