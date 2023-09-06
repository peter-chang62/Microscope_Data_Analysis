# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import clipboard as cr
import tables
from scipy.integrate import simpson
import os
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
import platform

figsize = np.array([4.64, 3.63])

folders = [i.name for i in os.scandir("I3_good_data/")]
if "DS_Store" in folders:
    folders.remove(".DS_Store")

# %% --------------------------------------------------------------------------
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


# %% --------------------------------------------------------------------------
figsize = np.array([4.64, 3.63])
file = tables.open_file("bio_sample_100GHz_fine.h5", "r")
# file = tables.open_file("bio_sample_100GHz_coarse.h5", "r")
data = file.root.data
absorbance = file.root.absorbance

pt_big_absorb = 145, 192
pt_less_absorb = 43, 222
pt_bckgnd = 195, 150

ppifg = 77760
resolution = 100
N = ppifg // resolution
N = N if N % 2 == 0 else N + 1

nu = np.fft.rfftfreq(N, d=1e-3) * ppifg
nu += nu[-1] * 2
wl = 299792458 / nu

fig_p, ax_p = plt.subplots(2, 1, figsize=figsize)
wnum = 1e4 / wl
ax_p[0].plot(
    wnum,
    -np.log(data[pt_big_absorb] / data[pt_bckgnd]),
    "C2",
)  # absorbance pt 2
ax_p[1].plot(
    wl,
    -np.log(data[pt_less_absorb] / data[pt_bckgnd]),
    "C3",
)  # absorbance pt 1
ax_p[0].set_xlim(1e4 / 3.267, 1e4 / 3.67)
ax_p[1].set_xlim(3.267, 3.67)
ax_p[0].set_ylim(0.439, 0.575)
ax_p[1].set_ylim(0.11, 0.309)
ax_p[0].spines.bottom.set_visible(False)
ax_p[1].spines.top.set_visible(False)
ax_p[0].xaxis.tick_top()
ax_p[0].xaxis.set_label_position("top")
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax_p[0].plot([0, 1], [0, 0], transform=ax_p[0].transAxes, **kwargs)
ax_p[1].plot([0, 1], [1, 1], transform=ax_p[1].transAxes, **kwargs)
ax_p[0].set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_p[1].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
fig_p.supylabel("absorbance", fontsize=10)
fig_p.tight_layout()
fig_p.subplots_adjust(hspace=0.1)  # adjust space between axes

img = absorbance[:, :, 97]
fig_f, ax_f = plt.subplots(1, 1)
x = np.arange(img.shape[1]) * 1.2
y = np.arange(img.shape[0]) * 1.2
ax_f.pcolormesh(
    x,
    y,
    img,
    vmin=0.15,  # fine
    vmax=0.66586,
    cmap="CMRmap_r_t",
)
ax_f.set_aspect("equal")
ax_f.axis(False)
# ax_f.plot(x[pt_bckgnd[1]], y[pt_bckgnd[0]], "o", color="C3")
ax_f.plot(x[pt_less_absorb[1]], y[pt_less_absorb[0]], "o", color="C3")
ax_f.plot(x[pt_big_absorb[1]], y[pt_big_absorb[0]], "o", color="C2")
scalebar = AnchoredSizeBar(
    ax_f.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper left",
    frameon=False,
    color="k",
    size_vertical=5 / 1.75,
)
ax_f.add_artist(scalebar)
fig_f.tight_layout()

# %% --------------------------------------------------------------------------
if platform.system().lower() == "darwin":
    path = r"/Users/peterchang/Resilio Sync/July ovarian FTIR I3/good data/"
elif platform.system().lower() == "windows":
    path = r"C:\\Users\\pchan\\Data\\July ovarian FTIR I3\\good data/"
x = np.memmap(path + folders[idx] + "/I3_Cropped", dtype="<f")
x.resize((394, *x_bp.shape))

# make sure to flip both! x_bp and x
x_bp_flip = x_bp[:, ::-1]
x = x[:, :, ::-1]

# check with imshow
plt.figure()
plt.imshow(x_bp_flip, vmin=vmin, vmax=vmax, cmap="CMRmap_r_t", interpolation="nearest")

# select point for spectrum
idx_x_ftir = 312
idx_y_ftir = 401

# %% --------------------------------------------------------------------------
wnum_ftir = np.genfromtxt("wnum.csv", delimiter=",")
wl_ftir = 1e4 / wnum_ftir

idx_overlap = np.logical_and(wnum_ftir > wnum.min(), wnum_ftir < wnum.max()).nonzero()

idx_x_dcs = 221
idx_y_dcs = 60

# uncomment to look at 2s averaged data
run_sample = np.load("run_sample.npy", mmap_mode="r")
run = np.load("run.npy", mmap_mode="r")
ft_b = run[-1]
ft_s = run_sample[-1]
t_b = np.fft.fftshift(np.fft.irfft(np.where(np.isnan(ft_b), 0, ft_b)))
t_s = np.fft.fftshift(np.fft.irfft(np.where(np.isnan(ft_s), 0, ft_s)))
resolution = 100
ppifg = 77760
center = ppifg // 2
n = ppifg // resolution
n = n if n % 2 == 0 else n + 1
t_b_a = t_b[center - n // 2 : center + n // 2]
t_s_a = t_s[center - n // 2 : center + n // 2]
ft_b_a = np.fft.rfft(np.fft.ifftshift(t_b_a))
ft_s_a = np.fft.rfft(np.fft.ifftshift(t_s_a))
curve2_2s = -np.log(abs(ft_s_a / ft_b_a))

curve1 = x[:, idx_y_ftir, idx_x_ftir][idx_overlap]
curve2 = absorbance[idx_y_dcs, idx_x_dcs]
curve2_gridded = interp1d(wnum, curve2, bounds_error=True)(wnum_ftir[idx_overlap])


def func(X, c1, c2):
    offst, mult = X
    return np.sum(abs(c1 * mult + offst - c2))


res = minimize(func, np.array([0, 0]), args=(curve1[26:54], curve2_gridded[26:54]))
curve1_scale = curve1 * res.x[1] + res.x[0]

fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.plot(wl_ftir[idx_overlap], curve1_scale, label="FTIR", color="C3")
ax.plot(wl, curve2 + 0.05, label="DCS 39 ms average", color="C1")
ax.plot(wl, curve2_2s + 0.05, label="DCS 2s average", color="C2")
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("absorbance (a.u.)")
ax.get_yaxis().set_ticks([])
ax_2 = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax.legend(loc="best")
fig.tight_layout()

# %% --------------------------------------------------------------------------
# find best match on the DCS side (smaller parameter space than FTIR side)
# error = np.zeros(img.shape)
# min_absb_dcs = 0.15
# for idx_y_dcs in tqdm(range(img.shape[0])):
#     for idx_x_dcs in range(img.shape[1]):
#         if img[idx_y_dcs, idx_x_dcs] < min_absb_dcs:
#             error[idx_y_dcs, idx_x_dcs] = np.nan
#             continue

#         curve1 = x[:, idx_y_ftir, idx_x_ftir][idx_overlap]
#         curve2_gridded = interp1d(
#             wnum,
#             absorbance[idx_y_dcs, idx_x_dcs],
#             bounds_error=True,
#         )(wnum_ftir[idx_overlap])

#         def func(X, c1, c2):
#             offst, mult = X
#             return np.sum(abs(c1 * mult + offst - c2))

#         res = minimize(
#             func,
#             np.array([0, 0]),
#             args=(curve1[26:54], curve2_gridded[26:54]),
#         )
#         error[idx_y_dcs, idx_x_dcs] = res.fun

# # %% -----
# plt.figure()
# plt.imshow(img, vmin=min_absb_dcs, cmap="CMRmap_r_t")

# # %% -----
# plt.figure()
# plt.hist(error.flatten(), bins=200)

# plt.figure()
# plt.imshow(error, cmap="CMRmap_r_t", vmax=2.8)
