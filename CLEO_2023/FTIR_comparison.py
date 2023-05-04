import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
import mkl_fft


def rfft(x, axis=-1):
    return mkl_fft.rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(mkl_fft.irfft_numpy(x, axis=axis), axes=axis)


c = 299792458
ppifg = 77760
center = ppifg // 2

path = r"/media/peterchang/Peter SSD/Research_Projects/Microscope/OvarianFTIR/I3/"

x = np.fromfile(path + "I3_output", dtype="<f")
x.resize(394, 1152, 3072)
wnum = np.genfromtxt("wnum.csv", delimiter=",")
wl = 1e4 / wnum

(ind,) = np.logical_and(3.3 < wl, wl < 3.6).nonzero()
(ind_full,) = np.logical_and(2.9 < wl, wl < 3.65).nonzero()

# %%
img = simpson(x[ind], axis=0)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img[:, ::-1][:, 1783:3071], cmap="cividis", vmin=0, vmax=10)
ax[1].imshow(
    np.load("../fig_commit/plot_data/coarse.npz")["data"][::-1][21:285, 16:309],
    cmap="cividis",
    vmax=47.5,
)
[i.axis(False) for i in ax]
fig.tight_layout()

# %% ----- plot the full absorption feature!

bckgnd = np.load("run.npy", mmap_mode="r")[-1].copy()
bio = np.load("run_sample.npy", mmap_mode="r")[-1].copy()
bckgnd[np.isnan(bckgnd)] = 0
bio[np.isnan(bio)] = 0

t_bckgnd = irfft(bckgnd)
t_bio = irfft(bio)
resolution = 50
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1
t_bckgnd_a = t_bckgnd[center - apod // 2 : center + apod // 2]
t_bio_a = t_bio[center - apod // 2 : center + apod // 2]
bckgnd_a = abs(rfft(t_bckgnd_a))
bio_a = abs(rfft(t_bio_a))
absorb = np.log(bckgnd_a / bio_a)
absorb[absorb > 0.4] = np.nan

# absorb = np.log(bckgnd / bio)

f = np.fft.rfftfreq(apod, d=1e-3) * ppifg
# f = np.fft.rfftfreq(ppifg, d=1e-3) * ppifg
f += f[-1] * 2
wl_comb = c / f
wnum_comb = f * 1e6 / (c / 1e-2)

(ind_comb,) = np.logical_and(2.9 < wl_comb, wl_comb < 3.65).nonzero()

# %%
fig, ax = plt.subplots(1, 1, figsize=np.array([4.64, 3.63]))
ax.plot(wl[ind_full], x[:, 749, 578][ind_full])
ax.plot(c / f[ind_comb], absorb[ind_comb] + 0.27)
ax.set_ylim(0.40074049646723675, 0.6840839957362752)
ax_2 = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax.set_ylabel("absorbance")
fig.tight_layout()

# %% ----- zoom in
pixel = np.load("../fig_commit/plot_data/pixel.npz")
fig_p, ax_p = plt.subplots(1, 1)
ax_p.plot(pixel["x"], pixel["y1"], color="C2")
ax_p.plot(pixel["x"], pixel["y2"], color="C3")
ax_p_2 = ax_p.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax_p_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_p.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_p.set_ylabel("absorbance")
ax_p.plot(wl[ind], x[:, 749, 578][ind])
fig_p.tight_layout()

# it looks like I'm 0.015 um off!
# I run with frep = 1e9, how far off can I be with that assumption?

dnu = c / 3.417e-6 - c / 3.43e-6

# mode number of the absorption feature
N_window = np.fft.rfftfreq(ppifg).size
N_feature = 9675  # absorption feature occurs at this index
N_mode = N_feature + N_window * 2

error = dnu / N_mode
print(f"means your lock was off from 1 GHz by {np.round(error * 1e-3, 3)} kHz")

# however, the error is on the scale of the FTIR's frequency resolution
dwnum = wnum[1] - wnum[0]
factor = dwnum * c / 1e-2 / dnu
