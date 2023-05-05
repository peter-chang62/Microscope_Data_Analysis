# %% ----- package imports & global stuff
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr


def window(fr, dfr):
    return fr**2 / (2 * dfr)


c = 299792458

wind_laser = c / 3.8e-6, c / 2.6e-6
wind_1ghz = window(1e9, 12.85e3)
wind_100MHz = window(100e6, 12.85e3)

cm = 1e-2
conversion = c / cm

nu = np.arange(0, wind_1ghz * 3, 1e9)

# %%
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(
    np.linspace(*[nu[0], nu[-1]], 100) * 1e-12,
    np.arange(2),
    np.zeros((2, 100)),
    cmap="gray",
    alpha=0.25,
)
[
    ax.axvline(wind_1ghz * i * 1e-12, color="k", linewidth=2, linestyle="--")
    for i in range(1, 4)
]
ax.pcolormesh(
    np.linspace(*[wind_laser[0], wind_laser[1]], 100) * 1e-12,
    np.arange(2),
    np.zeros((2, 100)),
    cmap="RdBu",
    alpha=0.5,
)
ax.set_xlabel("THz")
ax_2 = ax.secondary_xaxis(
    "top", functions=(lambda x: x * 1e12 / conversion, lambda x: x * conversion * 1e-12)
)
ax.get_yaxis().set_visible(False)
ax_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")

# %%
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(
    np.linspace(*[nu[0], nu[-1]], 100) * 1e-12,
    np.arange(2),
    np.zeros((2, 100)),
    cmap="gray",
    alpha=0.25,
)
[
    ax.axvline(wind_100MHz * i * 1e-12, color="k", linewidth=1, linestyle="--")
    for i in range(1, 301)
]
ax.pcolormesh(
    np.linspace(*[wind_laser[0], wind_laser[1]], 100) * 1e-12,
    np.arange(2),
    np.zeros((2, 100)),
    cmap="RdBu",
    alpha=0.5,
)
ax.set_xlabel("THz")
ax_2 = ax.secondary_xaxis(
    "top", functions=(lambda x: x * 1e12 / conversion, lambda x: x * conversion * 1e-12)
)
ax.get_yaxis().set_visible(False)
ax_2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
