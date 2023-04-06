import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
from tqdm import tqdm


# %% load the data, for some reason the first spectrum always looks wonky ...
# s_1 = np.load("temp/img1.npy", mmap_mode="r+")
# s_2 = np.load("temp/img2.npy", mmap_mode="r+")

# s_1 = s_1[1:]
# s_2 = s_2[1:]

# x = np.arange(0, s_1.shape[1] * 5, 5)
# y = np.arange(0, s_1.shape[0] * 5, 5)

# %% ------------------- f0 removal -------------------------------------------
# filt_s1 = np.array(
#     [
#         [19400, 19480],
#         [21500, 21750],
#         [25850, 26025],
#         [34480, 34560],
#         [0, 20],
#         [38860, 38900],
#     ]
# )

# filt_s2 = np.array(
#     [
#         [2410, 2450],
#         [19425, 19455],
#         [21537, 21753],
#         [25836, 26000],
#         [34440, 34540],
#         [0, 40],
#         [38860, 38890],
#     ]
# )


# def apply_filter(s, filt):
#     for f in filt:
#         s[f[0] : f[1]] = 0


# s_1.resize(s_1.shape[0] * s_1.shape[1], s_1.shape[2])

# for n, s in enumerate(tqdm(s_1)):
#     apply_filter(s, filt_s1)

# s_2.resize(s_2.shape[0] * s_2.shape[1], s_2.shape[2])
# for n, s in enumerate(tqdm(s_2)):
#     apply_filter(s, filt_s2)

# %% ---------------- combine images ------------------------------------------
# s_total = np.hstack(
#     (
#         s_1[:, :120],
#         s_2,
#     )
# )

# np.save("temp/s_total.npy", s_total)

# %% --------------- analyze combined image -----------------------------------
s_t = np.load("temp/s_total.npy", mmap_mode="r")

# ---------- single spectrum
# t = np.fft.fftshift(np.fft.irfft(s_t[s_t.shape[0] // 2, s_t.shape[1] // 2 + 0]))

# t_bckgnd = np.fft.fftshift(np.fft.irfft(s_t[0, 0]))
# t_bckgnd = np.fft.fftshift(np.fft.irfft(s_t[-1, -1]))
# t_bckgnd = np.load("avg_off_bio_sample.npy")

# ppifg = 77760
# center = ppifg // 2

# resolution = 50  # GHz
# apod = int(np.round(ppifg / resolution))
# if apod % 2 == 1:
#     apod += 1

# t = t[center - apod // 2 : center + apod // 2]
# ft = abs(np.fft.rfft(np.fft.ifftshift(t)))

# t_bckgnd = t_bckgnd[center - apod // 2 : center + apod // 2]
# ft_bckgnd = abs(np.fft.rfft(np.fft.ifftshift(t_bckgnd)))

# absorbance = -np.log(ft / ft_bckgnd)

# freq = np.fft.rfftfreq(len(t))
# (ind,) = np.logical_and(0.05 < freq, freq < 0.1912).nonzero()

# fig, ax = plt.subplots(1, 2, figsize=np.array([11.51, 4.8]))
# ax[0].plot(freq[ind], ft_bckgnd[ind], ".-")
# ax[0].plot(freq[ind], ft[ind], ".-")
# ax[1].plot(freq[ind], absorbance[ind])

# ---------- total image
ppifg = 77760
center = ppifg // 2

resolution = 1  # GHz
apod = int(np.round(ppifg / resolution))
if apod % 2 == 1:
    apod += 1

img = np.zeros((s_t.shape[0], s_t.shape[1], len(np.fft.rfftfreq(apod))))
shape_img = img.shape
shape_s = s_t.shape

t_bckgnd = np.fft.fftshift(np.fft.irfft(s_t[-1, -1]))
t_b = t_bckgnd[center - apod // 2 : center + apod // 2]
ft_b = abs(np.fft.rfft(np.fft.ifftshift(t_b)))

s_t.resize(s_t.shape[0] * s_t.shape[1], s_t.shape[2])
img.resize(img.shape[0] * img.shape[1], img.shape[2])
for n, s in enumerate(tqdm(s_t)):
    t = np.fft.fftshift(np.fft.irfft(s))
    t = t[center - apod // 2 : center + apod // 2]
    ft = abs(np.fft.rfft(np.fft.ifftshift(t)))
    absorbance = np.log(ft_b / ft)
    img[n] = absorbance

img.resize(*shape_img)
s_t.resize(*shape_s)


nu_grid = np.fft.rfftfreq(len(t), d=1e-9) * ppifg
nu_grid += nu_grid[-1] * 2
wl_grid = 299792458 / nu_grid * 1e6
wnum_grid = 1e4 / wl_grid

# %%
(ind,) = np.logical_and(3.1927 < wl_grid, wl_grid < 3.6914).nonzero()
fig, ax = plt.subplots(1, 1)
ax.plot(wnum_grid[ind], img[img.shape[0] // 2, img.shape[1] // 2][ind])
conversion = lambda x: 1e4 / x
ax2 = ax.secondary_xaxis("top", functions=(conversion, conversion))
ax2.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax.set_ylabel("absorbance")
fig.tight_layout()

# %%
# freq = np.fft.rfftfreq(len(t))
# i_1 = img[:, :, np.argmin(abs(freq - 0.12471))]
# i_2 = img[:, :, np.argmin(abs(freq - 0.09845))]
# i_3 = simpson(
#     img[:, :, np.argmin(abs(freq - 0.11647)) : np.argmin(abs(freq - 0.13521))], axis=-1
# )

# fig, ax = plt.subplots(1, 3, figsize=np.array([12.59, 4.8]))
# ax[0].imshow(i_1)
# ax[0].set_title("peak 1 height")
# ax[1].imshow(i_2)
# ax[1].set_title("peak 2 height")
# ax[2].imshow(i_3)
# ax[2].set_title("integrating over peak 1")
# fig.tight_layout()
