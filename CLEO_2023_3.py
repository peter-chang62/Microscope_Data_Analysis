# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import mkl_fft
from tqdm import tqdm


def rfft(x, axis=-1):
    return mkl_fft.rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(mkl_fft.irfft_numpy(x, axis=axis), axes=axis)


# %% -----
path = (
    r"/media/peterchang/Peter SSD/Research_Projects/"
    + r"Microscope/FreeRunningSpectra/11-09-2022/"
)
ppifg = 74180
center = ppifg // 2

# %% -----
# data = np.load(path + "stage1_5116_stage2_8500_53856x74180.npy")
data = np.load(path + "stage1_5300_stage2_8970_53856x74180.npy")
shape = data.shape
data.resize(data.size)
data = data[center:-center]
data.resize(shape[0] - 1, shape[1])

# %% -----
N_ifg = 100
data = data[: int(N_ifg * (data.shape[0] // N_ifg))]
shape = data.shape
data.resize(data.shape[0] // N_ifg, ppifg * N_ifg)

# optical
f = np.fft.rfftfreq(len(data[0]), d=1e-9) * ppifg * 1e-9
ind_d1GHz = int(1 / (f[1] - f[0]))

# rf
f_MHz = f / (ppifg * 1e-9) * 1e-6
df_MHz = f_MHz[1] - f_MHz[0]
ind_d1KHz = int(np.round(1e3 / (df_MHz * 1e6)))
ind_100KHz = abs(f_MHz - 0.1).argmin()

for n, x in enumerate(tqdm(data)):
    ft = rfft(x)
    ft_b = ft.copy()
    for i in range(-ind_d1KHz // 2, ind_d1KHz // 2 + 1):
        ft_b[ind_d1GHz + i :: ind_d1GHz] = 0
    ft_f = ft - ft_b
    ft_f[:ind_100KHz] = 0
    ft_f[abs(ft_f) > 1e7] = 0
    data[n] = irfft(ft_f)

data.resize(shape)

# %% -----
resolution = 50
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1
f_fit = np.fft.rfftfreq(apod)
ll, ul = 150, 400

ft_fit = rfft(data[:, center - apod // 2 : center + apod // 2])
p_fit = np.unwrap(np.angle(ft_fit))
f_c = f_fit[p_fit[0][ll:ul].argmin() + ll]
polyfit = np.polyfit((f_fit[ll:ul] - f_c), p_fit[:, ll:ul].T, deg=2)

# %% -----
f = np.fft.rfftfreq(ppifg)
index = np.arange(0, data.shape[0], 250)
index[-1] = data.shape[0]
for n in tqdm(range(len(index) - 1)):
    ft = rfft(data[index[n] : index[n + 1]])
    p = 0
    for m, coeff in enumerate(polyfit[::-1, index[n] : index[n + 1]]):
        p += (f - f_c) ** m * np.c_[coeff]
    ft *= np.exp(-1j * p)
    data[index[n] : index[n + 1]] = irfft(ft)

avg = np.mean(data, axis=0)
ft = abs(rfft(avg))
ind = np.array(
    [
        [0, 30],
        [18530, 18560],
        [20890, 20915],
        [20517, 20535],
        [34720, 34740],
        [37080, 37100],
    ]
)

for i in ind:
    ft[i[0] : i[1]] = np.nan
