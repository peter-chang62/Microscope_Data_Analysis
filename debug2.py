"""
this is a more detailed look at the f0 removal for 04-15-2023's background
stream. I don't like the spectrum changing shape while averaging down. This
happens around where f0's are located
"""
# %% ----- package imports
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mkl_fft
from tqdm import tqdm


def rfft(x, axis=-1):
    return mkl_fft.rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(mkl_fft.irfft_numpy(x, axis=axis), axes=axis)


# %% -----
path = r"/media/peterchang/Peter SSD/Research_Projects/Microscope/Images/04-15-2023/"
ppifg = 77760
center = ppifg // 2

# %% -----
data = np.load(path + "bckgnd_stream_25704x77760.npy")
shape = data.shape
data.resize(data.size)
data = data[center:-center]
data.resize(shape[0] - 1, shape[1])

# %% -----
N_ifg = 100
data = data[: int(N_ifg * (data.shape[0] // N_ifg))]
shape = data.shape
data.resize(data.shape[0] // N_ifg, ppifg * N_ifg)

f = np.fft.rfftfreq(len(data[0]), d=1e-9) * ppifg * 1e-9
f_MHz = f / (ppifg * 1e-9) * 1e-6
df_MHz = f_MHz[1] - f_MHz[0]
ind_d1KHz = int(np.round(1e3 / (df_MHz * 1e6)))
ind_100KHz = abs(f_MHz - 0.1).argmin()
ind_d1GHz = int(1 // (f[1] - f[0]))

for n, i in enumerate(tqdm(data)):
    ft = rfft(i)
    ft_b = ft.copy()
    for ii in range(-ind_d1KHz // 2, ind_d1KHz // 2 + 1):
        ft_b[ind_d1GHz + ii :: ind_d1GHz] = 0
    ft_f = ft - ft_b
    ft_f[:ind_100KHz] = 0
    data[n] = irfft(ft_f)

data.resize(shape)

# %% -----
resolution = 50
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1

ll, ul = 115, 363
f_fit = np.fft.rfftfreq(apod)
f = np.fft.rfftfreq(ppifg)

ft_fit = rfft(data[:, center - apod // 2 : center + apod // 2])
p_fit = np.unwrap(np.angle(ft_fit))
assert np.all(np.argmin(p_fit[:, ll:ul], axis=1) + ll < ul)
f_c = f_fit[p_fit[0][ll:ul].argmin() + ll]
polyfit = np.polyfit(f_fit[ll:ul] - f_c, p_fit[:, ll:ul].T, deg=2)

# %% -----
index = np.arange(0, data.shape[0], 250)
index[-1] = data.shape[0]
for n in tqdm(range(len(index) - 1)):
    ft = rfft(data[index[n] : index[n + 1]])
    p = 0
    for m, coeff in enumerate(polyfit[::-1, index[n] : index[n + 1]]):
        p += (f - f_c) ** m * np.c_[coeff]
    ft *= np.exp(-1j * p)
    data[index[n] : index[n + 1]] = irfft(ft)

# %% ----- running average
running_avg = np.zeros((data.shape[0], len(np.fft.rfftfreq(ppifg))))
avg = 0
print("calculating running average!")
for n, x in enumerate(tqdm(data)):
    avg = (avg * n + x) / (n + 1)
    running_avg[n] = abs(rfft(avg))

# %% -----
# np.save(path + "bckgnd_stream_fft_running_average.npy", running_avg)
stream = np.load(path + "bckgnd_stream_fft_running_average.npy")
absorb = stream[:, 4669:17646]
absorb = -np.log(absorb / absorb[-1])
snr = np.std(absorb, axis=1)
