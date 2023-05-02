# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import mkl_fft

# %%
path = r"D:\\Microscope\\FreeRunningSpectra\\11-09-2022/"
ppifg = 74180
center = ppifg // 2

# %%
bckgnd = np.load(path + "stage1_5116_stage2_8500_53856x74180.npy")
# bckgnd = np.load(path + "stage1_5300_stage2_8970_53856x74180.npy")
shape = bckgnd.shape
bckgnd.resize(bckgnd.size)
bckgnd = bckgnd[center:-center]
bckgnd.resize(shape[0] - 1, shape[1])

# %% ----- f0 removal
fft = mkl_fft.rfft_numpy(np.fft.ifftshift(np.mean(bckgnd[:250], axis=0)))

threshold = 150e3
step = 50
fft[:step] = 0
fft[-step:] = 0

(ind,) = (abs(fft) > threshold).nonzero()

ind = np.append(ind, np.array([2345, 2375, 32350, 32400]))

h = 0
dh = 250
while h < len(bckgnd):
    fft = mkl_fft.rfft_numpy(
        np.fft.ifftshift(bckgnd[h : h + dh], axes=1),
        axis=1,
    )
    fft[:, :step] = 0
    fft[:, -step:] = 0
    fft[:, 2345:2375] = 0
    fft[:, 32350:32400] = 0
    fft[:, 34700:34780] = 0
    fft[:, 4710:4730] = 0
    for i in ind:
        if i < step:
            fft[:, :step] = 0
        else:
            fft[:, i - step : i + step] = 0

    bckgnd[h : h + dh] = np.fft.fftshift(mkl_fft.irfft_numpy(fft, axis=1), axes=1)

    h += dh
    print(len(bckgnd) - h)

np.save("temp.npy", bckgnd)

# %%
bckgnd = np.load("temp.npy")

# %%
resolution = 100
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1

# %%
f_fit = np.fft.rfftfreq(apod)
ll, ul = 91, 178

h = 0
dh = 250
while h < len(bckgnd):
    fft_fit = mkl_fft.rfft_numpy(
        np.fft.ifftshift(
            bckgnd[h : h + dh, center - apod // 2 : center + apod // 2], axes=1
        ),
        axis=1,
    )
    p_fit = np.unwrap(np.angle(fft_fit))

    f_c = f_fit[p_fit[0][ll:ul].argmin() + ll]
    polyfit = np.polyfit((f_fit - f_c)[ll:ul], p_fit[:, ll:ul].T, deg=2)

    fft = mkl_fft.rfft_numpy(
        np.fft.ifftshift(bckgnd[h : h + dh], axes=1),
        axis=1,
    )
    f = np.fft.rfftfreq(ppifg)
    p = 0
    for n, coeff in enumerate(polyfit[::-1]):
        p += (f - f_c) ** n * np.c_[coeff]
    fft *= np.exp(-1j * p)

    bckgnd[h : h + dh] = np.fft.fftshift(mkl_fft.irfft_numpy(fft, axis=1), axes=1)
    h += dh
    print(len(bckgnd) - h)

avg = np.mean(bckgnd, axis=0)
fft_avg = abs(mkl_fft.rfft_numpy(np.fft.ifftshift(avg)))
