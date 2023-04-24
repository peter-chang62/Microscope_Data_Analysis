import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve

path = r"/Users/peterchang/Resilio Sync/OvarianFTIR/"
# path = r"temp/OvarianFTIR/"
data = np.fromfile(path + "D1", "<f")

data.shape = (394, 1280, 1280)
wnum = np.genfromtxt(path + "D1.hdr", skip_header=18, skip_footer=1, delimiter=",")
wl = 1e4 / wnum

wnum_resolution = np.mean(np.diff(wnum))
c = 299792458
cm = 1e-2
freq_resolution = wnum_resolution * c / cm

ppifg = 77760
center = ppifg // 2
resolution = 50
apod = int(np.round(ppifg / resolution))
if apod % 2 == 1:
    apod += 1

dcs_bckgnd = abs(
    np.fft.rfft(
        np.fft.ifftshift(
            np.load("avg_off_bio_sample.npy")[center - apod // 2 : center + apod // 2]
        )
    )
)
dcs_bio = abs(
    np.fft.rfft(
        np.fft.ifftshift(
            np.load("avg_on_bio_sample.npy")[center - apod // 2 : center + apod // 2]
        )
    )
)
absrptn = dcs_bio / dcs_bckgnd
absrbnc = -np.log(absrptn)
v_grid = np.fft.rfftfreq(apod, d=1e-9) * ppifg
v_grid += v_grid[-1] * 2
wl_grid = 299792458.0 / v_grid * 1e6
wnum_grid = 1e4 / wl_grid

# %%
(ind_dcs,) = np.logical_and(3.03 < wl_grid, wl_grid < 3.65).nonzero()
(ind_reddy,) = np.logical_and(3.03 < wl, wl < 3.65).nonzero()
fig, ax = plt.subplots(1, 1)
ax.plot(wl[ind_reddy], data[:, 1280 // 2, 1280 // 2][ind_reddy], ".-")
ax.plot(wl_grid[ind_dcs], absrbnc[ind_dcs])
conversion = lambda x: 1e4 / x
ax2 = ax.secondary_xaxis("top", functions=(conversion, conversion))
ax2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax.set_ylim(0, 0.4)
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
fig.tight_layout()

# %%
(ind,) = np.logical_and(2890 < wnum_grid, wnum_grid < 2950).nonzero()
ind_max = np.argmax(absrbnc[ind]) + ind.min()

# %%
ind_match = np.argmin(abs(wl - wl_grid[ind_max]))
plt.figure()
plt.title(
    "$\\mathrm{\\lambda}=$" + f"{np.round(wl[ind_match], 2)}" + " $\\mathrm{\\mu m}$"
)
plt.imshow(data[ind_match])

# %% convolve
# ---------- what's the real scan rate?

# ==================================
# You achieve the same effect if you just convolve it with a window of ones by
# the way. However, the image really just "blurs", it doesn't get smoother.
# That's likely something you would know if you were familiar with image
# processing.
# ==================================

vel = 5 * 12  # 5 um / pixel * 8 pixels / second
tau = 500 * ppifg / 1e9
blur = vel * tau
sptl_rsltn = 5 + blur

img_ftir = data[ind_match].copy()
step = int(np.round(sptl_rsltn / (1000 / 1280)))
size = np.asarray(img_ftir.shape) // step
img = np.zeros(size)
i = 0
j = 0
for n in range(img.shape[0]):
    i += step
    j = 0
    for m in range(img.shape[1]):
        section = img_ftir[i : i + step, j : j + step]
        img[n, m] = np.mean(section)
        j += step

plt.figure()
plt.title(
    "$\\mathrm{\\lambda}=$"
    + f"{np.round(wl[ind_match], 2)}"
    + " $\\mathrm{\\mu m}$"
    + "\n FTIR reference convolved to 7 $\\mathrm{\\mu m}$ resolution"
)
plt.imshow(img)
plt.tight_layout()

# %% diagnose
# fig, ax = plt.subplots(1, 1)
# save = False
# for n in tqdm(range(data.shape[0])):
#     ax.clear()
#     ax.set_title(
#         "$\\mathrm{\\lambda}=$" + f"{np.round(wl[n], 2)}" + " $\\mathrm{\\mu m}$"
#     )
#     ax.imshow(data[n])
#     fig.tight_layout()
#     if save:
#         plt.savefig(f"fig/{n}.png")
#     else:
#         plt.pause(0.05)
