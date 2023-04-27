# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
import os
from tqdm import tqdm
import mkl_fft


# ----------------------- image processing (done on 128 GB RAM computer) ------
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
# path = (
#     r"C:\\Users\\fastdaq\\SynologyDrive\\Research_Projects"
#     + r"\\Microscope\\Images\\04-15-2023\\"
# )
# path = r"H:\\Research_Projects\\Microscope\\Images\\04-15-2023/"
path = r"E:\\Research_Projects\\Microscope\\Images\\04-15-2023/"
coarse = np.load(path + "coarse/img_stacked_50GHz.npy")
fine = np.load(path + "fine/img_stacked_50GHz.npy")

coarse /= coarse[0, 0]
fine /= fine[-1, 0]
coarse = -np.log(coarse)
fine = -np.log(fine)

ppifg = 77760
tau_p = ppifg * 500 / 1e9
v_p = 12
factor = 1 + v_p * tau_p

# %%
fig_c, ax_c = plt.subplots(1, 1)
x = np.arange(coarse.shape[1]) * 5
y = np.arange(coarse.shape[0]) * 5
ax_c.pcolormesh(x, y, simpson(coarse[:, :, 140:240], axis=-1), cmap="cividis")
ax_c.set_aspect("equal")
ax_c.set_xlabel("$\\mathrm{\\mu m}$")
ax_c.set_ylabel("$\\mathrm{\\mu m}$")
# ax_c.set_title("integrated absorbance at 7.33 $\\mathrm{\\mu m}$ spatial sampling")
fig_c.tight_layout()

# %%
fig_f, ax_f = plt.subplots(1, 1)
x = np.arange(fine.shape[1]) * 1.2
y = np.arange(fine.shape[0]) * 1.2
ax_f.pcolormesh(x, y, simpson(fine[:, :, 140:240], axis=-1), cmap="cividis")
ax_f.set_aspect("equal")
ax_f.set_xlabel("$\\mathrm{\\mu m}$")
ax_f.set_ylabel("$\\mathrm{\\mu m}$")
# ax_f.set_title("integrated absorbance at 1.76 $\\mathrm{\\mu m}$ spatial sampling")
fig_f.tight_layout()

# %%
resolution = 50
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1
nu = np.fft.rfftfreq(apod, d=1e-3) * ppifg
nu += nu[-1] * 2
wl = 299792458 / nu
fig_p, ax_p = plt.subplots(1, 1)
(ind,) = np.logical_and(3.35 < wl, wl < 3.55).nonzero()
ax_p.plot(wl[ind], fine[0, -1][ind])
ax_p_2 = ax_p.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_p_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_p.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p.set_ylabel("absorbance")
# ax_p.set_title("Absorbance at 50 GHz")
fig_p.tight_layout()

# %% --------------------------- background stream ----------------------------
# path = r"H:\\Research_Projects\\Microscope\\Images\\04-15-2023/"
# stream = np.load(path + "bckgnd_stream_25704x77760.npy")
# shape = stream.shape
# stream.resize(stream.size)
# ppifg = 77760
# center = ppifg // 2
# stream = stream[center:-center]
# stream.resize((shape[0] - 1, ppifg))

# # %% ----- f0 removal

# # f0 was the same throughout
# ft = abs(mkl_fft.rfft_numpy(np.fft.ifftshift(stream[0])))
# (ind,) = (ft > 1.11e5).nonzero()
# step = 25
# FILT = np.zeros((len(ind), 2), dtype=int)
# for n, i in enumerate(ind):
#     if i < step:
#         filt = 0, step
#     else:
#         filt = i - step, i + step
#     FILT[n] = filt

# # after calculating the average, there's a small one here too... deal with it
# FILT = np.append(FILT, np.array([[19441 - step, 19441 + step]]), axis=0)

# h = 0
# step = 500
# FT = np.zeros((len(stream), len(np.fft.rfftfreq(ppifg))), dtype=np.complex128)
# while h < len(stream):
#     ft = mkl_fft.rfft_numpy(np.fft.ifftshift(stream[h : h + step], axes=1), axis=1)
#     for i in FILT:
#         ft[:, i[0] : i[1]] = 0
#     FT[h : h + step] = ft

#     stream[h : h + step] = np.fft.fftshift(mkl_fft.irfft_numpy(ft, axis=1), axes=1)

#     h += step
#     print(len(stream) - h)

# # %% ----- find center frequency for phase correction
# resolution = 500
# apod = ppifg // resolution
# if apod % 2 == 1:
#     apod += 1

# avg_500 = np.mean(stream[:500], axis=0)
# ft = mkl_fft.rfft_numpy(
#     np.fft.ifftshift(avg_500[center - apod // 2 : center + apod // 2])
# )
# f = np.fft.rfftfreq(apod)
# threshold = 10e3
# p = np.unwrap(np.angle(ft[abs(ft) > threshold]))
# f_p = f[abs(ft) > threshold]
# f_c = f_p[p.argmin()]
# f_p -= f_c

# # %% ----- phase correction
# h = 0
# step = 250
# while h < len(stream):
#     threshold = 10e3
#     ft = mkl_fft.rfft_numpy(
#         np.fft.ifftshift(
#             stream[h : h + step][:, center - apod // 2 : center + apod // 2], axes=1
#         ),
#         axis=1,
#     )
#     p = np.unwrap(np.angle(ft[:, abs(ft[0]) > threshold]))
#     f_p = f[abs(ft[0]) > threshold] - f_c
#     polyfit = np.polyfit(f_p, p.T, deg=2)
#     poly1d = 0
#     for n, i in enumerate(polyfit[::-1]):
#         poly1d += np.c_[i] * (np.fft.rfftfreq(ppifg) - f_c) ** n
#     FT[h : h + step] *= np.exp(-1j * poly1d)
#     stream[h : h + step] = np.fft.fftshift(
#         mkl_fft.irfft_numpy(FT[h : h + step], axis=1), axes=1
#     )

#     h += step
#     print(len(stream) - h)

# # %% ----- running averaging
# T = 0
# FT_avg = np.zeros(FT.shape)
# for n, t in enumerate(tqdm(stream)):
#     T = (T * n + t) / (n + 1)
#     FT_avg[n] = abs(mkl_fft.rfft_numpy(np.fft.ifftshift(T)))

# %% ----- plotting
# path = r"H:\\Research_Projects\\Microscope\\Images\\04-15-2023/"
path = r"E:\\Research_Projects\\Microscope\\Images\\04-15-2023/"
FT_avg = np.load(path + "bckgnd_stream_fft_running_average.npy")
ppifg = 77760
center = ppifg // 2

# %%
absorbance = -np.log(FT_avg / FT_avg[-1])
snr = np.std(absorbance[:, FT_avg[-1] > 15e3], axis=1)
t_ifg = ppifg / 1e9

# %%
nu = np.fft.rfftfreq(ppifg, d=1e-3) * ppifg
nu += nu[-1] * 2
wl = 299792458 / nu

# %%
fig_s, ax_s = plt.subplots(1, 1)
ax_s.plot(
    wl[FT_avg[-1] > 100],
    FT_avg[0][FT_avg[-1] > 100] / FT_avg[-1].max(),
    ".",
    markersize=1,
    label="single shot",
)
ax_s.plot(
    wl[FT_avg[-1] > 100],
    FT_avg[-1][FT_avg[-1] > 100] / FT_avg[-1].max(),
    ".",
    markersize=1,
    label="25,700 averages",
)
ax_s_2 = ax_s.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_s_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_s.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_s.set_ylabel("power spectral density")
ax_s.set_ylim(ymax=2)
# ax_s.set_title("Dual Comb Spectrum")
legend = ax_s.legend(loc="best", markerscale=10)
fig_s.tight_layout()

# %%
fig_snr, ax_snr = plt.subplots(1, 1)
ax_snr.loglog(np.arange(int(1e4)) * t_ifg, 1 / snr[: int(1e4)], "o")
ax_snr.set_xlabel("time (s)")
ax_snr.set_ylabel("snr")
# ax_snr.set_title("SNR")
fig_snr.tight_layout()

# %% ----- save all figures!
fig_c.savefig("fig_commit/coarse.png", dpi=300, transparent=True)
fig_f.savefig("fig_commit/fine.png", dpi=300, transparent=True)
fig_p.savefig("fig_commit/pixel.png", dpi=300, transparent=True)
fig_s.savefig("fig_commit/stream.png", dpi=300, transparent=True)
fig_snr.savefig("fig_commit/snr.png", dpi=300, transparent=True)
