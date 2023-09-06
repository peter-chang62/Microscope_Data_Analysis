# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import clipboard as cr
import tables
from scipy.integrate import simpson
import pynlo

figsize = np.array([4.64, 3.63])

# %% --------------------------------------------------------------------------
# file = tables.open_file("bio_sample_100GHz_fine.h5", "r")
file = tables.open_file("bio_sample_100GHz_coarse.h5", "r")
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

img = absorbance[:, :, 97]
fig_c, ax_c = plt.subplots(1, 1)
x = np.arange(img.shape[1]) * 5
y = np.arange(img.shape[0]) * 5
ax_c.pcolormesh(
    x,
    y,
    img,
    # vmin=0.15,  # fine
    vmin=0.19,  # coarse
    vmax=0.66586,
    cmap="CMRmap_r_t",
)
ax_c.set_aspect("equal")
ax_c.set_xlabel("$\\mathrm{\\mu m}$")
ax_c.set_ylabel("$\\mathrm{\\mu m}$")

scalebar = AnchoredSizeBar(
    ax_c.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "upper right",
    frameon=False,
    color="k",
    size_vertical=5,
)
ax_c.add_artist(scalebar)

fig_c.tight_layout()

# %% --------------------------------------------------------------------------
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

# fig_p, ax_p = plt.subplots(1, 1, figsize=figsize)
# norm = data[pt_bckgnd].max()
# # ax_p.plot(wl, data[pt_bckgnd] / norm, "C3")
# # ax_p.plot(wl, data[pt_less_absorb] / norm, "C2")
# # ax_p.plot(wl, data[pt_big_absorb] / norm, "C1")
# ax_p.plot(wl, -np.log(data[pt_less_absorb] / data[pt_bckgnd]), "C2")  # absorbance pt 1
# ax_p.plot(wl, -np.log(data[pt_big_absorb] / data[pt_bckgnd]), "C1")  # absorbance pt 2
# ax_p_2 = ax_p.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
# ax_p_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
# ax_p.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
# ax_p.set_ylabel("power spectral density (a.u.)")
# fig_p.tight_layout()

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
fig_f, ax_f = plt.subplots(1, 1, figsize=figsize)
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
file = tables.open_file("su8_sample_100GHz_coarse.h5", "r")
data = file.root.data
absorbance = file.root.absorbance

img = simpson(absorbance[:, :, 90:110], axis=-1)

fig_c_usaf, ax_c_usaf = plt.subplots(1, 1)
ppifg = 77760
tau_p = ppifg * 500 / 1e9
v_p = 10
factor = 1 + v_p * tau_p
x = np.arange(img.shape[1]) * 10
y = np.arange(img.shape[0]) * 10 / factor
ax_c_usaf.pcolormesh(y, x, img[::-1, ::-1].T, vmin=6.5, cmap="CMRmap_r_t")
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
    color="k",
    size_vertical=5,
)
ax_c_usaf.add_artist(scalebar)

fig_c_usaf.tight_layout()

# %% --------------------------------------------------------------------------
file = tables.open_file("su8_sample_100GHz_fine.h5", "r")
data = file.root.data
absorbance = file.root.absorbance

img = simpson(absorbance[:, :, 90:110], axis=-1)

pt_absorb = 54, 36
pt_absorb_2 = 52, 117
pt_bckgnd = 52, 78

ppifg = 77760
tau_p = ppifg * 500 / 1e9
v_p = 10
factor = 1 + v_p * tau_p

fig_f_usaf, ax_f_usaf = plt.subplots(1, 1)
x = np.arange(img.shape[1]) * 1.75
y = np.arange(img.shape[0]) * 1.75 / factor
ax_f_usaf.pcolormesh(x, y, img[::-1, ::-1], vmin=10, cmap="CMRmap_r_t")
ax_f_usaf.plot(x[::-1][pt_absorb[1]], y[::-1][pt_absorb[0]], "o", color="C2")
ax_f_usaf.plot(x[::-1][pt_absorb_2[1]], y[::-1][pt_absorb_2[0]], "o", color="C3")
ax_f_usaf.set_aspect("equal")
ax_f_usaf.axis(False)

scalebar = AnchoredSizeBar(
    ax_f_usaf.transData,
    100,
    "100 $\\mathrm{\\mu m}}$",
    "lower left",
    frameon=False,
    color="k",
    size_vertical=5 / 1.75,
)
ax_f_usaf.add_artist(scalebar)

fig_f_usaf.tight_layout()

resolution = 100
N = ppifg // resolution
N = N if N % 2 == 0 else N + 1

nu = np.fft.rfftfreq(N, d=1e-3) * ppifg
nu += nu[-1] * 2
wl = 299792458 / nu

norm = data[pt_bckgnd].max()
fig_p_usaf, ax_p_usaf = plt.subplots(1, 1, figsize=figsize)
# ax_p_usaf.plot(wl, data[pt_bckgnd] / norm, "C3")
# ax_p_usaf.plot(wl, data[pt_absorb] / norm, "C2")
ax_p_usaf.plot(wl, -np.log(data[pt_absorb] / data[pt_bckgnd]), "C2")  # absorbance plot
ax_p_usaf.plot(
    wl, -np.log(data[pt_absorb_2] / data[pt_bckgnd]), "C3"
)  # absorbance plot
ax_p_usaf.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p_usaf_2 = ax_p_usaf.secondary_xaxis(
    "top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x)
)
ax_p_usaf_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_p_usaf.set_ylabel("power spectral density (a.u.)")
fig_p_usaf.tight_layout()

# %% --------------------------------------------------------------------------
bckgnd = np.load("run.npy", mmap_mode="r")

ppifg = 77760
center = ppifg // 2
resolution = 100
N = ppifg // resolution
N = N if N % 2 == 0 else N + 1

nu = np.fft.rfftfreq(ppifg, d=1e-3) * ppifg
nu += nu[-1] * 2
wl = 299792458 / nu

nu_a = np.fft.rfftfreq(N, d=1e-3) * ppifg
nu_a += nu_a[-1] * 2
wl_a = 299792458 / nu_a

avg_500 = bckgnd[500].copy()
avg_500[np.isnan(avg_500)] = 0
t = np.fft.fftshift(np.fft.irfft(avg_500))
t = t[center - N // 2 : center + N // 2]
ft_a = np.fft.rfft(np.fft.ifftshift(t))

norm = np.nanmax(avg_500)

stream = np.load("../fig_commit/plot_data/stream.npz")
fig_s, ax_s = plt.subplots(1, 1, figsize=figsize)
ax_s.plot(stream["x"], stream["y1"], ".", markersize=1, label="single shot @ 1 GHz")
# ax_s.plot(wl, bckgnd[-1] / norm, ".", markersize=1, label="51,400 averages @ 1 GHz")
ax_s.plot(wl, bckgnd[bckgnd.shape[0] // 2] / norm, ".", markersize=1, label="25,700 averages @ 1 GHz")
ax_s.plot(wl_a, abs(ft_a) / norm, label="500 averages @ 100 GHz")
ax_s.set_ylim(ymax=2)
ax_s.legend(loc="best", markerscale=10)
ax_s.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_s.set_ylabel("power spectral density")
ax_s_2 = ax_s.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_s_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
fig_s.tight_layout()

snr = np.load("../fig_commit/plot_data/SNR_2D.npy")
resolution = np.logspace(np.log10(1), np.log10(100), num=100)
apod = ppifg // resolution
apod[apod % 2 == 1] += 1
apod = apod.astype(int)
resolution = ppifg / apod
n = np.arange(snr.shape[1]) + 1
tau = ppifg / 1e9

fig_r, ax_r = plt.subplots(1, 1, figsize=figsize)
img = ax_r.pcolormesh(
    n[10 : int(1e3)] * tau,
    resolution,
    1 / snr[:, 10 : int(1e3)],
    cmap="jet",
)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_xlabel("time (s)")
ax_r_2 = ax_r.secondary_xaxis("top", functions=(lambda x: x / tau, lambda x: x * tau))
ax_r_2.set_xlabel("# of averaged spectra")
ax_r.set_ylabel("resolution (GHz)")
colorbar = plt.colorbar(img, label="LOG SNR")
fig_r.tight_layout()

fig_allan, ax_allan = plt.subplots(1, 1, figsize=figsize)
ax_allan.loglog(
    n[10 : int(1e3)] * tau, snr[0, 10 : int(1e3)], "-", linewidth=3, label="1 GHz"
)
ax_allan.loglog(
    n[10 : int(1e3)] * tau, snr[25, 10 : int(1e3)], "-", linewidth=3, label="25 GHz"
)
ax_allan.loglog(
    n[10 : int(1e3)] * tau, snr[50, 10 : int(1e3)], "-", linewidth=3, label="50 GHz"
)
ax_allan.loglog(
    n[10 : int(1e3)] * tau, snr[99, 10 : int(1e3)], "-", linewidth=3, label="100 GHz"
)
ax_allan_2 = ax_allan.secondary_xaxis(
    "top", functions=(lambda x: x / tau, lambda x: x * tau)
)
ax_allan.set_xlabel("time (s)")
ax_allan_2.set_xlabel("# of averaged spectra")
ax_allan.set_ylabel("resolution (GHz)")
# ax_allan.axvline(500 * tau, color="k", linestyle="--")
ax_allan.legend(loc="best")
fig_allan.tight_layout()

# %% --------------------------------------------------------------------------
n_points = 2**11
min_wl = 800e-9
max_wl = 3e-6
center_wl = 1550e-9
t_fwhm = 50e-15
time_window = 20e-12
e_p = 3.5e-9

c = 299792458
pulse_g = pynlo.light.Pulse.Sech(
    n_points,
    c / max_wl,
    c / min_wl,
    c / center_wl,
    e_p,
    t_fwhm,
    min_time_window=time_window,
)
s_grat = np.genfromtxt("SPECTRUM_GRAT_PAIR.txt")
s_hnlf = np.genfromtxt("Spectrum_Stitched_Together_wl_nm.txt")
pulse_g.import_p_v(c / (s_grat[:, 0] * 1e-9), s_grat[:, 1], phi_v=None)

pulse_h = pulse_g.copy()
pulse_h.import_p_v(c / (s_hnlf[:, 0] * 1e-9), s_hnlf[:, 1], phi_v=None)

fig_grat, ax_grat = plt.subplots(1, 1)
ax_grat.plot(pulse_g.wl_grid * 1e6, pulse_g.p_v / pulse_g.p_v.max())
ax_grat.set_yticks([])
ax_grat.set_ylabel("linear intensity (a.u.)")
ax_grat.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_grat.spines["top"].set_visible(False)
ax_grat.spines["right"].set_visible(False)

fig_hnlf, ax_hnlf = plt.subplots(1, 1)
ax_hnlf.plot(pulse_h.wl_grid * 1e6, pulse_h.p_v / pulse_h.p_v.max())
ax_hnlf.set_ylabel("linear intensity (a.u.)")
ax_hnlf.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
