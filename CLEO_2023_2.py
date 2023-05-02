import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import numpy as np
import mkl_fft
from tqdm import tqdm

path = r"H:\\Research_Projects\\Shocktube\\static cell/"
# data = np.fromfile(path + "cell_with_mixture_10030x198850.bin", "<h")
data = np.load("temp.npy")
ppifg = 198850
center = ppifg // 2
# data = data[center:-center]
# data.resize(10030 - 1, ppifg)

# %% ----- f0 removal
# x = data[0].copy()
# ft = mkl_fft.rfft_numpy(np.fft.ifftshift(x))
# step = 50
# threshold = 300e3
# ft[:step] = 0
# ft[-step:] = 0
# (ind,) = (abs(ft) > threshold).nonzero()

# h = 0
# step = 250
# while h < len(data):
#     ft = mkl_fft.rfft_numpy(np.fft.ifftshift(data[h : h + step], axes=1), axis=1)
#     for i in ind:
#         if i < step:
#             ft[:, :step] = 0
#         else:
#             ft[:, i - step : i + step] = 0
#     data[h : h + step] = np.fft.fftshift(mkl_fft.irfft_numpy(ft))
#     h += step
#     print(len(data) - h)

# np.save("temp.npy", data)

# %% ----- prep phase correction
ft_100avg = abs(mkl_fft.rfft_numpy(np.fft.ifftshift(np.mean(data[:100], axis=0))))

resolution = 100
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1

# %%
# average 100 interferograms and apodize
ft = mkl_fft.rfft_numpy(
    np.fft.ifftshift(data[500][center - apod // 2 : center + apod // 2])
)

# skip to the spectrum (dfrep was not set to fastest possible here)
p = np.unwrap(np.angle(ft))
ll, ul = 764, 942
f = np.fft.rfftfreq(apod)

# plt.plot(f, abs(ft))
# plt.gca().twinx().plot(f, p, "C1")
# [plt.axvline(f[i]) for i in [ll, ul]]

f_c = f[p[ll:ul].argmin() + ll]
polyfit = np.polyfit((f - f_c)[ll:ul], p[ll:ul], deg=2)
p_fit = np.poly1d(polyfit)(f - f_c)

# plt.figure()
# plt.plot(f, p)
# plt.plot(f, p_fit)

# %%
h = 0
step = 250
f_full = np.fft.rfftfreq(ppifg)
while h < len(data):
    ft = mkl_fft.rfft_numpy(
        np.fft.ifftshift(
            data[h : h + step][:, center - apod // 2 : center + apod // 2], axes=1
        ),
        axis=1,
    )
    p = np.unwrap(np.angle(ft))
    polyfit = np.polyfit((f - f_c)[ll:ul], p[:, ll:ul].T, deg=2)

    p_fit = 0
    for n, coeff in enumerate(polyfit[::-1]):
        p_fit += (f_full - f_c) ** n * np.c_[coeff]
    ft = mkl_fft.rfft_numpy(
        np.fft.ifftshift(data[h : h + step], axes=1),
        axis=1,
    )
    ft *= np.exp(-1j * p_fit)
    data[h : h + step] = np.fft.fftshift(mkl_fft.irfft_numpy(ft, axis=1), axes=1)
    h += step
    print(len(data) - h)

avg = np.mean(data, 0)
ft_avg = abs(mkl_fft.rfft_numpy(np.fft.ifftshift(avg)))

# %%
nu = np.fft.rfftfreq(ppifg, d=1e-3) * ppifg
wl = 299792458 / nu
(ind,) = np.logical_and(3.02 < wl, wl < 5).nonzero()
plt.plot(
    wl[ind],
    ft_100avg[ind] / ft_avg[ind].max(),
    ".",
    markersize=1,
)
plt.plot(
    wl[ind],
    ft_avg[ind] / ft_avg[ind].max(),
    # ".",
    # markersize=1,
)
