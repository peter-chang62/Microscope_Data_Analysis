# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from tqdm import tqdm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.colors as colors


ppifg = 77760
center = ppifg // 2
figsize = np.array([4.64, 3.63])

# %% ----- stream -> I've moved this over to bio_static_reprocess.py
# # stream = np.load(
# #     r"/Volumes/Peter SSD/Research_Projects/Microscope/"
# #     + r"Images/04-15-2023/bckgnd_stream_fft_running_average.npy",
# #     mmap_mode="r",
# # )
# stream = np.load(
#     r"/media/peterchang/Peter SSD/Research_Projects/Microscope/"
#     + r"Images/04-15-2023/bckgnd_stream_fft_running_average.npy",
#     mmap_mode="r",
# )
# nu = np.fft.rfftfreq(ppifg, d=1e-3) * ppifg
# nu += nu[-1] * 2
# wl = 299792458 / nu
# norm = np.nanmax(stream[-1])
# fig_s, ax_s = plt.subplots(1, 1)
# ind = np.logspace(0, np.log10(len(stream)), dtype=int, num=100)
# ind[-1] = len(stream) - 1
# ind[0] = 0
# ind = np.append(np.arange(9), ind[ind > 8])
# t = np.round(ind * ppifg * 1e3 / 1e9, 2)
# save = False
# for n, ft in enumerate(tqdm(stream[ind])):
#     ax_s.clear()
#     ax_s.plot(wl[stream[-1] > 100], ft[stream[-1] > 100] / norm, ".", markersize=1)
#     # ax_s.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
#     # ax_s.set_ylabel("power spectral density")
#     # ax_s.set_title(f"{t[n]} ms")
#     ax_s.axis(False)
#     ax_s.set_ylim(ymax=1)
#     fig_s.tight_layout()
#     if save:
#         plt.savefig(f"fig/{n}.png", dpi=300, transparent=True)
#     else:
#         plt.pause(0.05)

# %% ----- coarse
coarse = np.load("../fig_commit/plot_data/coarse.npz")
fig_c, ax_c = plt.subplots(1, 1, figsize=figsize)
ax_c.pcolormesh(coarse["x"], coarse["y"], coarse["data"], cmap="cividis", vmax=47.5)
ax_c.plot(coarse["x"][154], coarse["y"][139], "o", color="C2")
ax_c.plot(coarse["x"][266], coarse["y"][167], "o", color="C3")
ax_c.set_xlabel("$\\mathrm{\\mu m}$")
ax_c.set_ylabel("$\\mathrm{\\mu m}$")
ax_c.set_aspect("equal")

# scalebar
scalebar = AnchoredSizeBar(
    ax_c.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper right",
    frameon=False,
    color="w",
    size_vertical=5,
)
ax_c.add_artist(scalebar)

fig_c.tight_layout()

# %% ----- fine
fine = np.load("../fig_commit/plot_data/fine.npz")
fig_f, ax_f = plt.subplots(1, 1, figsize=figsize)
ax_f.pcolormesh(fine["x"], fine["y"], fine["data"], cmap="cividis", vmax=45.7)
# ax_f.set_xlabel("$\\mathrm{\\mu m}$")
# ax_f.set_ylabel("$\\mathrm{\\mu m}$")
ax_f.axis(False)
ax_f.set_aspect("equal")

scalebar = AnchoredSizeBar(
    ax_f.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper left",
    frameon=False,
    color="w",
    size_vertical=5 / 1.75,
)
ax_f.add_artist(scalebar)

fig_f.tight_layout()

# %% ----- stream
stream = np.load("../fig_commit/plot_data/stream.npz")
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
# train = np.load(
#     r"/Volumes/Peter SSD/Research_Projects/Microscope/"
#     + r"Images/04-15-2023/bckgnd_stream_25704x77760.npy",
#     mmap_mode="r",
# )
# train = train[:25].copy()
# shape = train.shape
# train.resize(train.size)
# train = train[center:-center]
# train.resize((shape[0] - 1, shape[1]))
# t = np.arange(train.size) * 1e-3

# fig_t, ax_t = plt.subplots(1, 1, figsize=np.array([4.63, 1.43]))
# ax_t.plot(t, train.flatten())
# ax_t.set_xlabel("time ($\\mathrm{\\mu s}$)")
# ax_t.get_yaxis().set_visible(False)
# fig_t.tight_layout()

# # %% --- interferogram train gif
train = np.load(
    r"/Volumes/Peter SSD/Research_Projects/Microscope/"
    + r"Images/04-15-2023/bckgnd_stream_25704x77760.npy",
    mmap_mode="r",
)
shape = train.shape
train.resize(train.size)
train = train[center:-center]
train.resize((shape[0] - 1, shape[1]))

window = 50
t = np.arange(-window // 2, window // 2) * 1e-3

fig_t_g, ax_t_g = plt.subplots(1, 1, figsize=figsize)
N_ifg = 100
norm = train[:100].max()
vmax = 1.05
vmin = train[:100].min() / norm - 0.05
save = True
for n, ifg in enumerate(tqdm(train[:100])):
    ax_t_g.clear()
    ax_t_g.plot(
        t,
        ifg[center - window // 2 : center + window // 2] / norm, 'o--',
        linewidth=2,
    )
    ax_t_g.set_ylim(vmin, vmax)
    ax_t_g.axis(False)
    fig_t_g.tight_layout()
    if save:
        plt.savefig(f"../fig/{n}.png", dpi=300, transparent=True)
    else:
        plt.pause(0.01)

# %% ----- snr
snr = np.load("../fig_commit/plot_data/snr.npz")
fig_snr, ax_snr = plt.subplots(1, 1, figsize=figsize)
ax_snr.loglog(snr["x"], snr["y"], "o")
ax_snr.set_xlabel("time (s)")
ax_snr.set_ylabel("SNR")
tau = ppifg / 1e9
ax_snr_2 = ax_snr.secondary_xaxis(
    "top", functions=(lambda x: x / tau, lambda x: x * tau)
)
ax_snr_2.set_xlabel("# of averaged spectra")
fig_snr.tight_layout()
    
# %% ----- pixel
pixel = np.load("../fig_commit/plot_data/pixel.npz")
fig_p, ax_p = plt.subplots(1, 1, figsize=figsize)
ax_p.plot(pixel["x"], pixel["y1"], color="C2")
ax_p.plot(pixel["x"], pixel["y2"], color="C3")
ax_p_2 = ax_p.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_p_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_p.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p.set_ylabel("absorbance")
fig_p.tight_layout()

# %% ----- coarse usaf
fig_c_usaf, ax_c_usaf = plt.subplots(1, 1)
coarse_usaf = np.load("../fig_commit/plot_data/coarse_usaf.npz")
ax_c_usaf.pcolormesh(
    coarse_usaf["x"],
    coarse_usaf["y"],
    coarse_usaf["data"],
    cmap="cividis",
    vmax=160.5,
)
ax_c_usaf.plot(coarse_usaf["x"][39], coarse_usaf["y"][64], "o", color="C2")
ax_c_usaf.plot(coarse_usaf["x"][81], coarse_usaf["y"][113], "o", color="C3")
ax_c_usaf.set_aspect("equal")
ax_c_usaf.set_xlabel("$\\mathrm{\\mu m}$")
ax_c_usaf.set_ylabel("$\\mathrm{\\mu m}$")

# scalebar
scalebar = AnchoredSizeBar(
    ax_c_usaf.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper left",
    frameon=False,
    color="w",
    size_vertical=5,
)
ax_c_usaf.add_artist(scalebar)

fig_c_usaf.tight_layout()

# %% ----- fine usaf
fig_f_usaf, ax_f_usaf = plt.subplots(1, 1)
fine_usaf = np.load("../fig_commit/plot_data/fine_usaf.npz")
ax_f_usaf.pcolormesh(
    fine_usaf["x"],
    fine_usaf["y"],
    fine_usaf["data"],
    cmap="cividis",
    vmax=168.93,
)
ax_f_usaf.set_aspect("equal")
ax_f_usaf.set_xlabel("$\\mathrm{\\mu m}$")
ax_f_usaf.set_ylabel("$\\mathrm{\\mu m}$")
ax_f_usaf.axis(False)

scalebar = AnchoredSizeBar(
    ax_f_usaf.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "lower left",
    frameon=False,
    color="w",
    size_vertical=5 / 1.75,
)
ax_f_usaf.add_artist(scalebar)

fig_f_usaf.tight_layout()

# %% ---- pixel usaf
fig_p_usaf, ax_p_usaf = plt.subplots(1, 1, figsize=figsize)
pixel_usaf = np.load("../fig_commit/plot_data/pixel_usaf.npz")
ax_p_usaf.plot(pixel_usaf["x"], pixel_usaf["y1"], color="C2")
ax_p_usaf.plot(pixel_usaf["x"], pixel_usaf["y2"], color="C3")
ax_p_usaf.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p_usaf.set_ylabel("absorbance")
ax_p_usaf_2 = ax_p_usaf.secondary_xaxis(
    "top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x)
)
ax_p_usaf_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
fig_p_usaf.tight_layout()

# %%
supercontinuum = np.genfromtxt(
    "../fig_commit/plot_data/Spectrum_Stitched_Together_wl_nm.txt"
)
fig_cont, ax_cont = plt.subplots(1, 1)
ax_cont.plot(supercontinuum[:, 0], supercontinuum[:, 1])
ax_cont.get_yaxis().set_visible(False)
ax_cont.set_xlabel("wavelength (nm)")
fig_cont.tight_layout()

# %% ----- static cell
fig_stat, ax_stat = plt.subplots(1, 1)
static_cell = np.load("../fig_commit/plot_data/static_cell.npy")
ax_stat.plot(static_cell[:, 0], static_cell[:, 1] / static_cell[:, 1].max())
ax_stat.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_stat.set_ylabel("power spectral density")
ax_stat.spines["right"].set_visible(False)
ax_stat.spines["top"].set_visible(False)
fig_stat.tight_layout()

fig_stat_zoom, ax_stat_zoom = plt.subplots(1, 1)
static_cell = np.load("../fig_commit/plot_data/static_cell.npy")
(ind,) = np.logical_and(4.385 < static_cell[:, 0], static_cell[:, 0] < 4.404).nonzero()
ax_stat_zoom.plot(
    static_cell[:, 0][ind], static_cell[:, 1][ind] / static_cell[:, 1].max()
)
ax_stat_zoom.axis(False)
fig_stat_zoom.tight_layout()

# %% ----- on and off su8
fig_su8, ax_su8 = plt.subplots(1, 1, figsize=np.array([3.69, 2.71]))
bckgnd_su8 = np.load("../fig_commit/plot_data/bckgnd_for_su8.npz")
su8 = np.load("../fig_commit/plot_data/su8.npz")
ax_su8.plot(bckgnd_su8["x"], bckgnd_su8["y"] / np.nanmax(bckgnd_su8["y"]))
ax_su8.plot(su8["x"], su8["y"] / np.nanmax(bckgnd_su8["y"]))
ax_su8.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_su8.set_ylabel("power spectral density")
fig_su8.tight_layout()

# %% ----- on and off bio sample
fig_bio, ax_bio = plt.subplots(1, 1, figsize=np.array([3.69, 2.71]))
bckgnd_bio = np.load("../fig_commit/plot_data/bckgnd_for_bio.npz")
bio = np.load("../fig_commit/plot_data/bio.npz")
ax_bio.plot(bckgnd_bio["x"], bckgnd_bio["y"] / np.nanmax(bio["y"]))
ax_bio.plot(bio["x"], bio["y"] / np.nanmax(bio["y"]))
ax_bio.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_bio.set_ylabel("power spectral density")
fig_bio.tight_layout()

# %% ----- apodization speed analysis
snr = np.load("../fig_commit/plot_data/SNR_2D.npy")
resolution = np.logspace(np.log10(1), np.log10(100), num=100)
apod = ppifg // resolution
apod[apod % 2 == 1] += 1
apod = apod.astype(int)
resolution = ppifg / apod
n = np.arange(snr.shape[1]) + 1
tau = ppifg / 1e9

fig_r, ax_r = plt.subplots(1, 1, figsize=figsize)
ax_r.pcolormesh(
    n[10 : int(1e3)] * tau,
    resolution,
    1 / snr[:, 10 : int(1e3)],
    cmap="cividis",
)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_xlabel("time (s)")
ax_r_2 = ax_r.secondary_xaxis("top", functions=(lambda x: x / tau, lambda x: x * tau))
ax_r_2.set_xlabel("# of averaged spectra")
ax_r.set_ylabel("resolution (GHz)")
fig_r.tight_layout()

fig_r_s, ax_r_s = plt.subplots(1, 1, figsize=figsize)
ax_r_s.loglog(
    n[:-1][10 : int(1e3)] * tau,
    1 / snr[0][:-1][10 : int(1e3)],
    "o",
    label="1 GHz",
)
ind_5 = abs(resolution - 5).argmin()
ax_r_s.loglog(
    n[:-1][10 : int(1e3)] * tau,
    1 / snr[ind_5][:-1][10 : int(1e3)],
    "o",
    label=f"{int(np.round(resolution[ind_5]))} GHz",
)
ind_50 = abs(resolution - 50).argmin()
ax_r_s.loglog(
    n[:-1][10 : int(1e3)] * tau,
    1 / snr[ind_50][:-1][10 : int(1e3)],
    "o",
    label=f"{int(np.round(resolution[ind_50]))} GHz",
)
ax_r_s.legend(loc="best")
ax_r_s.set_xlabel("time (s)")
ax_r_s_2 = ax_r_s.secondary_xaxis(
    "top", functions=(lambda x: x / ppifg, lambda x: x * ppifg)
)
ax_r_s_2.set_xlabel("# of averaged spectra")
ax_r_s.set_ylabel("SNR")
fig_r_s.tight_layout()
