# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard as cr
import imageio.v3 as iio
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import convolve

figsize = np.array([4.64, 3.63])


def lightness(img):
    img_no_alpha = img[:, :, :3]
    return 0.5 * (img_no_alpha.max(axis=2) + img_no_alpha.min(axis=2))


def luminosity(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return 0.21 * r + 0.72 + g + 0.07 * b


def average(img):
    img_no_alpha = img[:, :, :3]
    return np.sum(img_no_alpha, axis=2) / 3


folder = "I3_good_data_reddy_imgs/"
names = [i.name for i in os.scandir(folder)]
# %% --------------------------------------------------------------------------
for idx in range(len(names)):
    # im = iio.imread("I3_good_data_reddy_imgs/wp_ov-63_hd_16ca_2850_band.png")
    im = iio.imread(folder + names[idx])

    lit = lightness(im)
    lum = luminosity(im)
    avg = average(im)

    lit /= lit.max()
    lum /= lum.max()
    avg /= avg.max()

    # the luminosity looks the most like the original :)
    # %%
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # img = ax.imshow(lum[:, ::-1], cmap="CMRmap_t")
    img = ax.imshow(im[:, ::-1], cmap="CMRmap_t")
    s = names[idx].split(".png")[0]
    fig.suptitle(f"{s}: {im.shape[0]}x{im.shape[1]}")

    # scalebar = AnchoredSizeBar(
    #     ax.transData,
    #     100,
    #     "100 $\\mathrm{\\mu m}}$",
    #     "upper right",
    #     frameon=False,
    #     color="k",
    #     size_vertical=5,
    # )
    # ax.add_artist(scalebar)

    fig.tight_layout()

    # plt.savefig("clipboard.png", transparent=True, dpi=300)

# %% ----------------------- just plot one ------------------------------------
# idx = 6  # or 7
im = iio.imread(folder + "wp_ov-65_sd_16ca_2850_band.png")
# im = iio.imread(folder + names[idx])

lit = lightness(im)
lum = luminosity(im)
avg = average(im)

lit /= lit.max()
lum /= lum.max()
avg /= avg.max()

# the luminosity looks the most like the original :)
fig, ax = plt.subplots(1, 1)
lum = convolve(lum, np.ones((2, 2)), mode="valid")
x = np.arange(lum.shape[1]) * 5.5 / 2
y = np.arange(lum.shape[0]) * 5.5 / 2
img = ax.pcolormesh(x, y, lum[::-1, ::-1], cmap="CMRmap_t")
ax.set_aspect("equal")
scalebar = AnchoredSizeBar(
    ax.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper right",
    frameon=False,
    color="k",
    size_vertical=5,
)
ax.add_artist(scalebar)
ax.axis(False)
fig.tight_layout()

# %% ----------------------- just plot one ------------------------------------
# idx = 4  # or 5
im = iio.imread(folder + "wp_ov-65_hd_1ca_2850_band.png")
# im = iio.imread(folder + names[idx])

lit = lightness(im)
lum = luminosity(im)
avg = average(im)

lit /= lit.max()
lum /= lum.max()
avg /= avg.max()

# zoom in
fig, ax = plt.subplots(1, 1, figsize=figsize)
x = np.arange(lum.shape[1]) * 1.1
y = np.arange(lum.shape[0]) * 1.1
(idx_x,) = np.logical_and(164.52 < x, x < 709.22).nonzero()
(idx_y,) = np.logical_and(1000.0 < y, y < 1250.15).nonzero()
img = ax.pcolormesh(x[idx_x], y[idx_y], lum[::-1, ::-1][idx_y][:, idx_x], cmap="CMRmap_t")
ax.set_aspect("equal")
scalebar = AnchoredSizeBar(
    ax.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper left",
    frameon=False,
    color="k",
    size_vertical=5 / 1.75,
)
ax.add_artist(scalebar)
ax.axis(False)
fig.tight_layout()

# plt.savefig("clipboard.png", transparent=True, dpi=300)
