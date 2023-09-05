# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import clipboard as cr
import tables
from scipy.integrate import simpson

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
