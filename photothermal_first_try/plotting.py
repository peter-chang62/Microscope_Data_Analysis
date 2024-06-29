# %% ----- package imports
import clipboard
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import ifftshift, rfft, rfftfreq
from scipy.constants import c

path = r"/Volumes/Peter SSD 2/photothermal_v2_1/unpacked/averaged/"

# x1 and x2 have polystyrene spectra
x1 = np.load(path + "MokuDataLoggerData_20240329_120520.npy")
x2 = np.load(path + "MokuDataLoggerData_20240329_122819.npy")

wnum_to_ghz = lambda wnum: wnum * c * 100 / 1e9
window = (
    lambda ppifg, wnum: int(ppifg // wnum_to_ghz(wnum))
    + int(ppifg // wnum_to_ghz(wnum)) % 2
)

# %% -----
# fig, ax = plt.subplots(1, 1)
# ppifg = x1.shape[0]
# center = ppifg // 2
# ax.plot(
#     x1[:, 0][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
#     x1[:, 1][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
# )
# ax.plot(
#     x1[:, 0][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
#     x1[:, 2][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
# )

# %% -----
fig, ax = plt.subplots(1, 1)
ppifg = x2.shape[0]
center = ppifg // 2
ax.plot(
    x2[:, 0][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
    x2[:, 1][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
)
ax.plot(
    x2[:, 0][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
    x2[:, 2][center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2],
)

# %% -----
fig, ax = plt.subplots(1, 1)
ppifg = x2.shape[0]
center = ppifg // 2
t = x2[:, 0]
dt = t[1] - t[0]
freq = rfftfreq(window(ppifg, 5), dt)
nu = freq * dt / 1e-9 * ppifg
wnum = nu / c / 100
xlim = (2004.8660119308915, 4014.6701174872287)
(idx,) = np.logical_and(wnum > xlim[0], wnum < xlim[1]).nonzero()
ax.plot(
    wnum[idx],
    abs(
        rfft(
            ifftshift(
                x2[:, 1][
                    center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2
                ]
            )
        )
    )[idx],
)
ax.plot(
    wnum[idx],
    abs(
        rfft(
            ifftshift(
                x2[:, 2][
                    center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2
                ]
            )
        )
    )[idx],
)
ax_2 = ax.secondary_xaxis(
    "top",
    functions=(
        lambda x: (x * c * 100) / dt * 1e-9 / ppifg * 1e-3,
        lambda x: (x * dt / 1e-9 * ppifg) / c / 100,
    ),
)
ax.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_2.set_xlabel("RF modulation frequency (kHz)")

# %% -----
fig, ax = plt.subplots(1, 1)
ppifg = x2.shape[0]
center = ppifg // 2
xlim = (2698.2577903339834, 3221.912117163706)
(idx,) = np.logical_and(wnum > xlim[0], wnum < xlim[1]).nonzero()
ax.plot(
    wnum[idx],
    abs(
        rfft(
            ifftshift(
                x2[:, 2][
                    center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2
                ]
            )
        )
    )[idx]
    / abs(
        rfft(
            ifftshift(
                x2[:, 1][
                    center - window(ppifg, 5) // 2 : center + window(ppifg, 5) // 2
                ]
            )
        )
    )[idx],
)
ax_2 = ax.secondary_xaxis(
    "top",
    functions=(
        lambda x: (x * c * 100) / dt * 1e-9 / ppifg * 1e-3,
        lambda x: (x * dt / 1e-9 * ppifg) / c / 100,
    ),
)
ax.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_2.set_xlabel("RF modulation frequency (kHz)")
