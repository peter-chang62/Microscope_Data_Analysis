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
path = r"/media/peterchang/Peter SSD/Research_Projects/Microscope/Images/04-15-2023/"
ppifg = 77760
center = ppifg // 2

# %% -----
data = np.load(path + "bckgnd_stream_25704x77760.npy")
shape = data.shape
data.resize(data.size)
data = data[center:-center]
data.resize(shape[0] - 1, shape[1])

index = np.arange(0, data.shape[0], 250)
index[-1] = data.shape[0]

# %% ----- where is f0?
ind = np.array(
    [
        [18068, 24577],
        [25416, 25708],
        [34482, 34509],
    ]
)

# %% ----- f0 removal
print("removing f0!")
for n in tqdm(range(len(index) - 1)):
    ft = rfft(data[index[n] : index[n + 1]])
    ft[:, :250] = 0
    ft[:, -250:] = 0
    for i in ind:
        ft[:, i[0] : i[1]] = 0
    data[index[n] : index[n + 1]] = irfft(ft)

# %% -----
resolution = 50
apod = ppifg // resolution
apod = apod if apod % 2 == 0 else apod + 1

# %% -----
ll, ul = 146, 302
f_fit = np.fft.rfftfreq(apod)
f = np.fft.rfftfreq(ppifg)

ft_fit = rfft(data[:, center - apod // 2 : center + apod // 2])
p_fit = np.unwrap(np.angle(ft_fit))
assert np.all(np.argmin(p_fit[:, ll:ul], axis=1) + ll < ul)
f_c = f_fit[p_fit[0][ll:ul].argmin() + ll]
polyfit = np.polyfit(f_fit[ll:ul] - f_c, p_fit[:, ll:ul].T, deg=2)

# %% ----- don't remove f0!
# data = np.load(path + "bckgnd_stream_25704x77760.npy")
# shape = data.shape
# data.resize(data.size)
# data = data[center:-center]
# data.resize(shape[0] - 1, shape[1])

print("phase correcting!")
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

for i in ind:
    running_avg[:, i[0] : i[1]] = np.nan

np.save(path + "bckgnd_stream_fft_running_average.npy", running_avg)
