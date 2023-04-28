# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from tqdm import tqdm

ppifg = 77760
center = ppifg // 2

# %% ----- stream
# stream = np.load(
#     r"/Volumes/Peter SSD/Research_Projects/Microscope/"
#     + r"Images/04-15-2023/bckgnd_stream_fft_running_average.npy",
#     mmap_mode="r",
# )
# nu = np.fft.rfftfreq(ppifg, d=1e-3) * ppifg
# nu += nu[-1] * 2
# wl = 299792458 / nu
# norm = stream[-1].max()
# fig_s, ax_s = plt.subplots(1, 1)
# ind = np.logspace(0, np.log10(len(stream)), dtype=int, num=100)
# ind[-1] = len(stream) - 1
# ind[0] = 0
# ind = np.append(np.arange(9), ind[ind > 8])
# t = np.round(ind * ppifg * 1e3 / 1e9, 2)
# save = True
# for n, ft in enumerate(tqdm(stream[ind])):
#     ax_s.clear()
#     ax_s.plot(wl[stream[-1] > 100], ft[stream[-1] > 100] / norm, ".", markersize=1)
#     # ax_s.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
#     # ax_s.set_ylabel("power spectral density")
#     # ax_s.set_title(f"{t[n]} ms")
#     ax_s.axis(False)
#     ax_s.set_ylim(ymax=1.5)
#     fig_s.tight_layout()
#     if save:
#         plt.savefig(f"fig/{n}.png", dpi=300, transparent=True)
#     else:
#         plt.pause(0.05)

# %% ----- coarse
figsize = np.array([4.64, 3.63])

coarse = np.load("fig_commit/plot_data/coarse.npz")
fig_c, ax_c = plt.subplots(1, 1, figsize=figsize)
ax_c.pcolormesh(coarse["x"], coarse["y"], coarse["data"], cmap="cividis")
ax_c.set_xlabel("$\\mathrm{\\mu m}$")
ax_c.set_ylabel("$\\mathrm{\\mu m}$")
ax_c.set_aspect("equal")
fig_c.tight_layout()

# %% ----- fine
fine = np.load("fig_commit/plot_data/fine.npz")
fig_f, ax_f = plt.subplots(1, 1, figsize=figsize)
ax_f.pcolormesh(fine["x"], fine["y"], fine["data"], cmap="cividis")
# ax_f.set_xlabel("$\\mathrm{\\mu m}$")
# ax_f.set_ylabel("$\\mathrm{\\mu m}$")
ax_f.axis(False)
ax_f.set_aspect("equal")
fig_f.tight_layout()

# %% ----- stream
stream = np.load("fig_commit/plot_data/stream.npz")
fig_s, ax_s = plt.subplots(1, 1, figsize=figsize)
ax_s.plot(stream["x"], stream["y1"], ".", markersize=1, label="single shot")
ax_s.plot(stream["x"], stream["y2"], ".", markersize=1, label="25,700 averages")
ax_s.set_ylim(ymax=2)
ax_s.legend(loc="best", markerscale=10)
ax_s.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_s.set_ylabel("power spectral density")
ax_s_2 = ax_s.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_s_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
fig_s.tight_layout()

# %% ----- interferogram train
train = np.load(
    r"/Volumes/Peter SSD/Research_Projects/Microscope/"
    + r"Images/04-15-2023/bckgnd_stream_25704x77760.npy",
    mmap_mode="r",
)
train = train[:25].copy()
shape = train.shape
train.resize(train.size)
train = train[center:-center]
train.resize((shape[0] - 1, shape[1]))
t = np.arange(train.size) * 1e-3

fig_t, ax_t = plt.subplots(1, 1, figsize=np.array([4.63, 1.43]))
ax_t.plot(t, train.flatten())
ax_t.set_xlabel("time ($\\mathrm{\\mu s}$)")
ax_t.get_yaxis().set_visible(False)
fig_t.tight_layout()

# %% ----- snr
snr = np.load("fig_commit/plot_data/snr.npz")
fig_snr, ax_snr = plt.subplots(1, 1, figsize=figsize)
ax_snr.loglog(snr["x"], snr["y"], "o")
ax_snr.set_xlabel("time (s)")
ax_snr.set_ylabel("absorbance snr")
tau = ppifg / 1e9
ax_snr_2 = ax_snr.secondary_xaxis(
    "top", functions=(lambda x: x / tau, lambda x: x * tau)
)
ax_snr_2.set_xlabel("# of averaged spectra")
fig_snr.tight_layout()

# %% ----- pixel
pixel = np.load("fig_commit/plot_data/pixel.npz")
fig_p, ax_p = plt.subplots(1, 1, figsize=figsize)
ax_p.plot(pixel["x"], pixel["y"])
ax_p_2 = ax_p.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_p_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_p.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p.set_ylabel("absorbance")
fig_p.tight_layout()

# %% ----- coarse usaf
fig_c_usaf, ax_c_usaf = plt.subplots(1, 1)
coarse_usaf = np.load("fig_commit/plot_data/coarse_usaf.npz")
ax_c_usaf.pcolormesh(
    coarse_usaf["x"], coarse_usaf["y"], coarse_usaf["data"], cmap="cividis"
)
ax_c_usaf.set_aspect("equal")
ax_c_usaf.set_xlabel("$\\mathrm{\\mu m}$")
ax_c_usaf.set_ylabel("$\\mathrm{\\mu m}$")
fig_c_usaf.tight_layout()

# %% ----- fine usaf
fig_f_usaf, ax_f_usaf = plt.subplots(1, 1)
fine_usaf = np.load("fig_commit/plot_data/fine_usaf.npz")
ax_f_usaf.pcolormesh(fine_usaf["x"], fine_usaf["y"], fine_usaf["data"], cmap="cividis")
ax_f_usaf.set_aspect("equal")
ax_f_usaf.set_xlabel("$\\mathrm{\\mu m}$")
ax_f_usaf.set_ylabel("$\\mathrm{\\mu m}$")
ax_f_usaf.axis(False)
fig_f_usaf.tight_layout()

# %% ---- pixel usaf
fig_p_usaf, ax_p_usaf = plt.subplots(1, 1, figsize=figsize)
pixel_usaf = np.load("fig_commit/plot_data/pixel_usaf.npz")
ax_p_usaf.plot(pixel_usaf["x"], pixel_usaf["y"])
ax_p_usaf.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p_usaf.set_ylabel("absorbance")
ax_p_usaf_2 = ax_p_usaf.secondary_xaxis(
    "top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x)
)
ax_p_usaf_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
fig_p_usaf.tight_layout()
