# %% -----
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from numpy.fft import fftshift, ifftshift, rfft, irfft, rfftfreq
from tqdm import tqdm
import blit
import clipboard

path = r"/Volumes/Peter SSD 2/alazar_data/"

# %% -----
with open(path + "SU8_05-03-2024.npy", "rb") as f:
    data = np.load(f, allow_pickle=True)
# with open("Glomerulus_05-03-2024_4.npy", "rb") as f:
#     data = np.load(f, allow_pickle=True)
pt = data["pt"][0].astype(np.int16)
mir = data["mir"][0].astype(np.int16)

# %% -----
level = 2500
(idx,) = (mir > level).nonzero()
spacing = np.diff(idx)
ppifg = int(np.round(spacing[spacing > 1.5e5].mean()))
ppifg = ppifg if ppifg % 2 == 0 else ppifg + 1
center = ppifg // 2

start = mir[:ppifg].argmax()
start = start - center if start > center else start + center
mir = mir[start:]
pt = pt[start:]

N = mir.size // ppifg
mir = mir[: N * ppifg]
pt = pt[: N * ppifg]
mir.resize((N, ppifg))
pt.resize((N, ppifg))

# %% -----
for n in range(mir.shape[0]):
    roll = center - mir[n].argmax()
    mir[n] = np.roll(mir[n], roll)
    pt[n] = np.roll(pt[n], roll)

# %% -----
resolution = 5 * 100 * c / 1e9
window = ppifg // resolution
window = int(window if window % 2 == 0 else window + 1)
freq = rfftfreq(window)

mir = mir[:, center - window // 2 : center + window // 2]
pt = pt[:, center - window // 2 : center + window // 2]

# %% -----
for n in range(mir.shape[0]):
    mir[n] -= np.round(mir[n].mean()).astype(int)
    pt[n] -= np.round(pt[n].mean()).astype(int)

# %% -----
idx = 0
ft = rfft(ifftshift(mir[idx]))
p = np.unwrap(np.angle(ft))
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()
ax.plot(freq, abs(ft))
ax2.plot(freq, p, "C1")

# %% -----
xlim = (0.011126352971804247, 0.012518190351686387)
(idx_xlim,) = np.logical_and(xlim[0] < freq, freq < xlim[1]).nonzero()

degree = 2
for idx in tqdm(range(0, mir.shape[0])):
    # ----- calculate phase correction
    ft = rfft(ifftshift(mir[idx]))
    p = np.unwrap(np.angle(ft))
    polyfit = np.polyfit(freq[idx_xlim], p[idx_xlim], deg=degree)
    poly1d = np.poly1d(polyfit)
    p_corr = -poly1d(freq)

    # ----- apply phase correction to channel 2
    ft = rfft(ifftshift(mir[idx]))
    ft *= np.exp(1j * p_corr)
    igm = fftshift(irfft(ft))

    mir[idx] = fftshift(irfft(ft))

    # ----- apply phase correction to channel 1
    ft = rfft(ifftshift(pt[idx]))
    ft *= np.exp(1j * p_corr)
    pt[idx] = fftshift(irfft(ft))

    if idx == 2:
        fig, ax = plt.subplots(1, 1)
        (l1,) = ax.plot(
            mir[idx][window // 2 - 2859 : window // 2 + 2859], animated=True
        )
        (l2,) = ax.plot(
            mir[idx][window // 2 - 2859 : window // 2 + 2859], animated=True
        )
        bm = blit.BlitManager(fig.canvas, [l1, l2])
        bm.update()
    elif idx > 2:
        l2.set_ydata(mir[idx][window // 2 - 2859 : window // 2 + 2859])
        bm.update()

# %% -----
avg_mir = np.mean(mir, axis=0)
avg_pt = np.mean(pt, axis=0)

# %% ----- plotting
nu = rfftfreq(window, d=1 / (1e6)) * 1e9 / 1e6 * ppifg
wnum = nu / c / 100
xlim = (1923.6660816460897, 4144.224286717351)
(idx_xlim,) = np.logical_and(xlim[0] < wnum, wnum < xlim[1]).nonzero()
ft_ch2 = abs(rfft(ifftshift(avg_mir)))
ft_ch1 = abs(rfft(ifftshift(avg_pt)))

figsize = np.array([3.96, 2.53])

fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.plot(wnum[idx_xlim], ft_ch2[idx_xlim] / ft_ch2[idx_xlim].max(), label="DCS")
ax.plot(wnum[idx_xlim], ft_ch1[idx_xlim] / ft_ch2[idx_xlim].max(), label="PT DCS")
ax2 = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
ax.legend(loc="best")
ax.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax2.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=figsize)
xlim = 2675.5075224742577, 3072.7742518153927
(idx_xlim,) = np.logical_and(xlim[0] < wnum, wnum < xlim[1]).nonzero()
ax.plot(wnum[idx_xlim], ft_ch1[idx_xlim] / ft_ch2[idx_xlim], label="PT DCS")
ax2 = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / x, lambda x: 1e4 / x))
# ax.legend(loc="best")
ax.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax2.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=figsize)
t = np.arange(-window // 2, window // 2) * 1 / 1e6 * 1e3
xlim = (-3, 3)
(idx_xlim,) = np.logical_and(xlim[0] < t, t < xlim[1]).nonzero()
ax.plot(t[idx_xlim], avg_mir[idx_xlim] / avg_mir[idx_xlim].max(), label="DCS")
ax.plot(t[idx_xlim], avg_pt[idx_xlim] / avg_mir[idx_xlim].max(), label="PT DCS")
ax.legend(loc="best")
ax.set_xlabel("ms")
fig.tight_layout()
