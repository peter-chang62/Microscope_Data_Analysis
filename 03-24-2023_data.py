# %% packge imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from tqdm import tqdm
from scipy.integrate import simpson
import scipy.constants as sc


# %%  -------------------------------------------------------------------------
# def apply_filter(freq, ft, filt):
#     for f in filt:
#         ll, ul = f
#         (ind,) = np.logical_and(ll < freq, freq < ul).nonzero()
#         ft[ind] = 0


# path = r"H:\\Research_Projects\\Microscope\\FreeRunningSpectra\\03-23-2023/"
# data = np.load(path + "think_off_bio_sample_51408x77760.npy", mmap_mode="r")
# N_ifg, ppifg = data.shape
# center = ppifg // 2
# data.resize(data.size)
# data = data[center:-center]
# data.resize((N_ifg - 1, ppifg))

# filt = np.array(
#     [
#         [-1, 15],
#         [249.8, 250.2],
#         [275, 290],
#         [330.3, 336.3],
#         [436.8, 437.2],
#         [499, 501],
#     ]
# )

# # _oversrite = np.save("_overwrite.npy", np.zeros(data.shape))
# _overwrite = np.load("_overwrite.npy", "r+")

# apod = 400

# %%  -------------------------------------------------------------------------
# N = 4
# n = np.arange(0, len(data), len(data) // 8)
# n[-1] = len(data)
# start = n[:-1]
# end = n[1:]

# # console = 4
# # for n in tqdm(range(start[console], end[console])):
# #     x = data[n]
# #     x = np.roll(x, center - np.argmax(x))

# #     ft = np.fft.rfft(np.fft.ifftshift(x))
# #     f_MHz = np.fft.rfftfreq(ppifg, d=1e-9) * 1e-6
# #     apply_filter(f_MHz, ft, filt)
# #     x = np.fft.fftshift(np.fft.irfft(ft))

# #     x_a = x[center - apod // 2 : center + apod // 2]
# #     ft_a = np.fft.rfft(np.fft.ifftshift(x_a))
# #     p_a = np.unwrap(np.angle(ft_a))
# #     f_a = np.fft.rfftfreq(len(x_a), d=1e-9) * 1e-6

# #     # fig, ax = plt.subplots(1, 1, num="phase")
# #     # ax_p = ax.twinx()
# #     # ax_p.plot(f_a, p_a, color="C1")
# #     # ax.plot(f_a, abs(ft_a), color="C0")

# #     # obtained from plotting
# #     f_c = 148
# #     f_ll, f_ul = 81, 240
# #     (ind,) = np.logical_and(f_ll < f_a, f_a < f_ul).nonzero()

# #     f_centered = f_a - f_c
# #     polyfit = np.polyfit(f_centered[ind], p_a[ind], deg=2)
# #     poly1d = np.poly1d(polyfit)
# #     ft *= np.exp(-1j * poly1d(f_MHz - f_c))
# #     x_c = np.fft.fftshift(np.fft.irfft(ft))

# #     _overwrite[n] = x_c

# # print(f"console # {console}")

# -----------------------------------------------------------------------------
# avg = 0
# for n, x in enumerate(tqdm(_overwrite)):
#     avg = (avg * n + x) / (n + 1)

# np.save("avg_off_bio_sample.npy", avg)

# %% --------------------------------------------------------------------------
bio = abs(np.fft.rfft(np.fft.ifftshift(np.load("avg_on_bio_sample.npy"))))
bckgnd = abs(np.fft.rfft(np.fft.ifftshift(np.load("avg_off_bio_sample.npy"))))

ppifg = 77760
center = ppifg // 2

T = bio / bckgnd
absorbance = -np.log(T)

v_grid = np.fft.rfftfreq(ppifg, d=1e-9) * ppifg
v_grid += v_grid[-1] * 2
wl_grid = sc.c * 1e6 / v_grid

# %%
fig, ax = plt.subplots(1, 2, num="Reddy bio sample", figsize=np.array([13.68, 4.8]))
ax[0].plot(wl_grid, bckgnd / bckgnd.max())
ax[0].plot(wl_grid, bio / bckgnd.max())
ax[1].plot(wl_grid, absorbance)
ax[1].set_xlim(3.35, 3.55)
ax[1].set_ylim(0.16, 0.31)
ax[0].set_ylabel("power")
ax[1].set_ylabel("absorbance")
[i.set_xlabel("wavelength ($\\mathrm{\\mu m}$)") for i in ax]
fig.tight_layout()

# %% --------------------------------------------------------------------------
img = np.fromfile(r"/Users/peterchang/Resilio Sync/OvarianFTIR/D1", "<f")
img.resize(394, 1280, 1280)
wl = (
    np.genfromtxt(
        r"/Users/peterchang/Resilio Sync/OvarianFTIR/D1.hdr",
        skip_header=18,
        skip_footer=1,
        delimiter=",",
    )
    * 1e-3
)

# %%
ax[0].plot(wl, img[:, 1280 // 2, 1280 // 2])

# %%
plt.figure()
plt.imshow(simpson(img, axis=0))
