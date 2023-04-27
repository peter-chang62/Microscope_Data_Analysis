# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
import os
from tqdm import tqdm
import mkl_fft

# %% ----- coarse
# path = r"D:\\Microscope\\Images\\04-20-2023/"
# coarse = np.load(path + "img_10um.npy")
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

# resolution = 10
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

# np.save(path + "img_10um_10GHz.npy", coarse_apod)

# %% ----- fine
# path = r"D:\\Microscope\\Images\\04-20-2023\\fine/"
# fine = [np.load(path + f"img{i}.npy") for i in range(1, 4)]
# fine[0] = fine[0][:, :89]
# fine = np.hstack(fine)
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

# resolution = 10
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

# np.save(path + "img_stacked_10GHz.npy", fine_apod)

# %% ----- plotting
# path = r"D:\\Microscope\\Images\\04-20-2023/"
path = r"E:\\Research_Projects\\Microscope\\Images\\04-20-2023/"
coarse = np.load(path + "img_10um_50GHz.npy")
fine = np.load(path + "fine/img_stacked_50GHz.npy")

abs_coarse = -np.log(coarse / coarse[0, 0])
abs_fine = -np.log(fine / fine[0, 0])

ppifg = 77760
center = ppifg // 2
tau_p = ppifg * 500 / 1e9
v_p = 10
factor = 1 + v_p * tau_p

# %%
fig_c, ax_c = plt.subplots(1, 1)
y = np.arange(coarse.shape[0]) * 10 / factor
x = np.arange(coarse.shape[1]) * 10
ax_c.pcolormesh(y, x, simpson(abs_coarse[:, :, 100:300]).T[::-1, ::-1], cmap="cividis")
ax_c.set_aspect("equal")
ax_c.set_xlabel("$\\mathrm{\\mu m}$")
ax_c.set_ylabel("$\\mathrm{\\mu m}$")
fig_c.tight_layout()

# %%
fig_f, ax_f = plt.subplots(1, 1)
y = np.arange(fine.shape[0]) * 1.75 / factor
x = np.arange(fine.shape[1]) * 1.75
ax_f.pcolormesh(x, y, simpson(abs_fine[:, :, 100:300][::-1, ::-1]), cmap="cividis")
ax_f.set_aspect("equal")
ax_f.set_xlabel("$\\mathrm{\\mu m}$")
ax_f.set_ylabel("$\\mathrm{\\mu m}$")
fig_f.tight_layout()

# %%
fig_p, ax_p = plt.subplots(1, 1)
resolution = 50
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1
nu = np.fft.rfftfreq(apod, d=1e-3) * ppifg
nu += nu[-1] * 2
wl = 299792458 / nu
(ind,) = np.logical_and(3.25 < wl, wl < 3.65).nonzero()
ax_p.plot(wl[ind], abs_fine[::-1, ::-1][85, 24][ind])
ax_p_2 = ax_p.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_p.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_p.set_ylabel("absorbance")
fig_p.tight_layout()

# %% ----- save all figures
fig_c.savefig("fig_commit/coarse_usaf.png", dpi=300, transparent=True)
fig_f.savefig("fig_commit/fine_usaf.png", dpi=300, transparent=True)
