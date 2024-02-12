"""
This time apodize down to 100 GHz and just take the peak. Another thing I
figured out here is how to initialize an appendable buffer on disk :)
"""

# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard as cr
from scipy.integrate import simpson
from tqdm import tqdm
import tables
import os

try:
    import mkl_fft

    rfft_numpy = mkl_fft.rfft_numpy
    irfft_numpy = mkl_fft.irfft_numpy
except ImportError:
    rfft_numpy = np.fft.rfft
    irfft_numpy = np.fft.irfft


# %% ----- function defs
def rfft(x, axis=-1):
    return rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(irfft_numpy(x, axis=axis), axes=axis)


# %% ----- generate 100 GHz image
# path = r"/Volumes/Peter SSD/Research_Projects/Microscope/Images/04-15-2023/coarse/"
# resolution = 50
# ppifg = 77760
# center = ppifg // 2

# N = ppifg // resolution
# N = N if N % 2 == 0 else N + 1
# N_f = np.fft.rfftfreq(N).size

# file = tables.open_file("_overwrite.h5", "w")
# atom = tables.Float64Atom()
# names = [i.name for i in os.scandir(path) if "img" in i.name]
# key = lambda s: float(s.split("img")[1].split(".npy")[0])
# names.sort(key=key)

# array_initialized = False
# for i in names:
#     data = np.load(path + i, mmap_mode="r")
#     data = data[2:]
#     shape = data.shape

#     if not array_initialized:
#         array = file.create_earray(
#             file.root,
#             "data",
#             atom=atom,
#             shape=(shape[0], 0, N_f),
#         )
#         array_initialized = True

#     threshold = 36e3
#     for ii in tqdm(range(shape[1])):
#         x = data[:, ii].copy()
#         x[x > threshold] = 0
#         t = irfft(x)
#         t = t[:, center - N // 2 : center + N // 2]
#         x_a = abs(rfft(t))
#         array.append(x_a[:, np.newaxis, :])

# # calculate absorbance
# array_absorbance = file.create_earray(
#     file.root,
#     "absorbance",
#     atom=atom,
#     shape=(shape[0], 0, N_f),
# )
# for i in tqdm(range(array.shape[1])):
#     x = array[:, i]
#     array_absorbance.append(-np.log(x / array[0, 0])[:, np.newaxis, :])

# file.close()

# %% ----- look at results!
# file = tables.open_file("bio_sample_100GHz_fine.h5", "r")
file = tables.open_file("bio_sample_100GHz_coarse.h5", "r")
data = file.root.data
absorbance = file.root.absorbance

# %% ----- plot a bunch of histograms
# for i in range(60, 120):
#     fig, ax = plt.subplots(1, 1)
#     ax.hist(absorbance[:, :, i].flatten(), bins=250)
#     ax.set_title(i)

# %% ----- plot images with a bunch of vmins
# save = True
# fig, ax = plt.subplots(1, 1)
# for n, i in enumerate(tqdm(np.arange(0.02, 0.3, 0.01))):
#     ax.clear()
#     ax.pcolormesh(
#         absorbance[:, :, 97],
#         # vmin=0.2046,
#         vmin=i,
#         vmax=0.6586,
#         cmap="cividis",
#     )
#     ax.set_title(i)
#     ax.set_aspect("equal")
#     if save:
#         plt.savefig(f"../fig/{n}.png", dpi=300, transparent=True)

# %% -----
# plt.figure()
# plt.pcolormesh(
#     absorbance[:, :, 97],
#     # vmin=0.14,  # for fine
#     vmin=0.2046,  # for coarse
#     vmax=0.6586,
#     cmap="cividis",
# )
# plt.gca().set_aspect("equal")

# %% ----- save figure data!
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

figsize = np.array([4.64, 3.63])
fig, ax = plt.subplots(1, 1, figsize=figsize)
norm = data[pt_bckgnd].max()
ax.plot(wl, data[pt_bckgnd] / norm, "C3")
ax.plot(wl, data[pt_less_absorb] / norm, "C2")
ax.plot(wl, data[pt_big_absorb] / norm, "C1")
ax_2 = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("power spectral density (a.u.)")
fig.tight_layout()

img = absorbance[:, :, 97]
fig, ax = plt.subplots(1, 1)
x = np.arange(img.shape[1]) * 1.2
y = np.arange(img.shape[0]) * 1.2
ax.pcolormesh(
    x,
    y,
    img,
    # vmin=0.15,  # fine
    vmin=0.19,  # coarse
    vmax=0.66586,
    cmap="cividis",
)
ax.set_aspect("equal")
ax.set_xlabel("$\\mathrm{\\mu m}$")
ax.set_ylabel("$\\mathrm{\\mu m}$")
ax.plot(x[pt_bckgnd[1]], y[pt_bckgnd[0]], 'o', color="C3")
ax.plot(x[pt_less_absorb[1]], y[pt_less_absorb[0]], 'o', color="C2")
ax.plot(x[pt_big_absorb[1]], y[pt_big_absorb[0]], 'o', color="C1")
fig.tight_layout()
