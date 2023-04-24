# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
import os
from tqdm import tqdm
import mkl_fft

# %% ----- coarse
# path = (
#     r"C:\\Users\\fastdaq\\SynologyDrive\\Research_Projects"
#     + r"\\Microscope\\Images\\04-15-2023\\coarse/"
# )
# names = [i.name for i in os.scandir(path) if "img" in i.name and "stack" not in i.name]
# coarse = np.hstack([np.load(path + i) for i in names])
# coarse = coarse[2:]

# step = 20
# shape = coarse.shape
# coarse.shape = shape[0] * shape[1], shape[2]
# for n, s in enumerate(tqdm(coarse)):
#     s[:step] = 0
#     s[-step:] = 0
#     (ind,) = (s > 35e3).nonzero()
#     for i in ind:
#         if i < step:
#             s[:step] = 0
#         else:
#             s[i - step : i + step] = 0
# coarse.shape = shape

# resolution = 50
# ppifg = 77760
# center = ppifg // 2
# apod = ppifg // resolution
# if apod % 2 == 1:
#     apod += 1

# shape = coarse.shape
# coarse.shape = shape[0] * shape[1], shape[2]
# coarse_apod = np.zeros((shape[0] * shape[1], len(np.fft.rfftfreq(apod))))
# for n, s in enumerate(tqdm(coarse)):
#     t = np.fft.fftshift(mkl_fft.irfft_numpy(s))
#     t = t[center - apod // 2 : center + apod // 2]
#     coarse_apod[n] = mkl_fft.rfft_numpy(np.fft.ifftshift(t))
# coarse.shape = shape
# coarse_apod.shape = shape[0], shape[1], len(np.fft.rfftfreq(apod))

# np.save(path + "img_stacked_50GHz.npy", coarse_apod)

# %% ----- fine
# path = (
#     r"C:\\Users\\fastdaq\\SynologyDrive\\Research_Projects\\"
#     + r"Microscope\\Images\\04-15-2023\\fine/"
# )
# fine = np.hstack([np.load(path + f"img{i}.npy") for i in range(1, 10)])
# fine = fine[2:]

# step = 20
# shape = fine.shape
# fine.shape = shape[0] * shape[1], shape[2]
# for n, s in enumerate(tqdm(fine)):
#     s[:step] = 0
#     s[-step:] = 0
#     (ind,) = (s > 35e3).nonzero()
#     for i in ind:
#         if i < step:
#             s[:step] = 0
#         else:
#             s[i - step : i + step] = 0
# fine.shape = shape

# resolution = 50
# ppifg = 77760
# center = ppifg // 2
# apod = ppifg // resolution
# if apod % 2 == 1:
#     apod += 1

# shape = fine.shape
# fine.shape = shape[0] * shape[1], shape[2]
# fine_apod = np.zeros((shape[0] * shape[1], len(np.fft.rfftfreq(apod))))
# for n, s in enumerate(tqdm(fine)):
#     t = np.fft.fftshift(mkl_fft.irfft_numpy(s))
#     t = t[center - apod // 2 : center + apod // 2]
#     fine_apod[n] = mkl_fft.rfft_numpy(np.fft.ifftshift(t))
# fine.shape = shape
# fine_apod.shape = shape[0], shape[1], len(np.fft.rfftfreq(apod))

# np.save(path + "img_stacked_50GHz.npy", fine_apod)

# # %% ----- plotting
path = (
    r"C:\\Users\\fastdaq\\SynologyDrive\\Research_Projects"
    + r"\\Microscope\\Images\\04-15-2023\\"
)
coarse = np.load(path + "coarse/img_stacked_50GHz.npy")
fine = np.load(path + "fine/img_stacked_50GHz.npy")

coarse /= coarse[0, 0]
fine /= fine[-1, 0]
coarse = -np.log(coarse)
fine = -np.log(fine)

# %%
fig_c, ax_c = plt.subplots(1, 1, num="coarse")
x = np.arange(coarse.shape[1]) * 5
y = np.arange(coarse.shape[0]) * 5
ax_c.pcolormesh(x, y, simpson(coarse[:, :, 140:240], axis=-1), cmap='cividis')
ax_c.set_aspect("equal")
ax_c.set_xlabel("$\\mathrm{\\mu m}$")
ax_c.set_ylabel("$\\mathrm{\\mu m}$")
fig_c.tight_layout()

# %%
fig_f, ax_f = plt.subplots(1, 1, num="fine")
x = np.arange(fine.shape[1]) * 1.2
y = np.arange(fine.shape[0]) * 1.2
ax_f.pcolormesh(x, y, simpson(fine[:, :, 140:240], axis=-1), cmap='cividis')
ax_f.set_aspect("equal")
ax_f.set_xlabel("$\\mathrm{\\mu m}$")
ax_f.set_ylabel("$\\mathrm{\\mu m}$")
fig_f.tight_layout()
