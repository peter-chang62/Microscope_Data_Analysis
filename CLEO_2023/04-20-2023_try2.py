"""
Really the only difference in try 2 is processing on 8 GB of RAM
using _overwrite.h5 and pytables. And the fact that the image is generated at
100 GHz frequency resolution, and the absorption peak is used as the
distinguishing factor
"""

# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import tables
from tqdm import tqdm
from scipy.integrate import simpson
import os

try:
    import mkl_fft

    rfft_numpy = mkl_fft.rfft_numpy
    irfft_numpy = mkl_fft.irfft_numpy
except ImportError:
    rfft_numpy = np.fft.rfft
    irfft_numpy = np.fft.irfft


# %% ----- function defs
def rfft(x, axis=-1):
    return rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(irfft_numpy(x, axis=axis), axes=axis)


# %% -----
path = r"/Volumes/Peter SSD/Research_Projects/Microscope/Images/04-20-2023/"

# coarse
data = np.load(path + r"img_10um.npy", mmap_mode="r")
data = data[2:]
DATA = [data]

# fine
# DATA = [np.load(path + f"fine/img{i}.npy", mmap_mode="r") for i in range(1, 4)]
# DATA[0] = DATA[0][:, :89]  # cut off at 89th linescan for first img1.npy
# DATA = [i[2:] for i in DATA]

ppifg = 77760
center = ppifg // 2

threshold = 25e3

resolution = 50
N = ppifg // resolution
N = N if N % 2 == 0 else N + 1

file = tables.open_file("_overwrite.h5", "w")
atom = tables.Float64Atom()

array_initialized = False
for data in DATA:
    for i in tqdm(range(data.shape[1])):
        if not array_initialized:
            array = file.create_earray(
                file.root,
                "data",
                atom=atom,
                shape=(data.shape[0], 0, np.fft.rfftfreq(N).size),
            )
            array_initialized = True

        x = data[:, i].copy()
        x[x > threshold] = 0

        t = irfft(x)
        t = t[:, center - N // 2 : center + N // 2]
        x_a = abs(rfft(t))

        array.append(x_a[:, np.newaxis, :])

array_absorbance = file.create_earray(
    file.root,
    "absorbance",
    atom=atom,
    shape=(data.shape[0], 0, np.fft.rfftfreq(N).size),
)

for i in tqdm(range(array.shape[1])):
    x = array[:, i]
    array_absorbance.append(-np.log(x / array[0, 0])[:, np.newaxis, :])

file.close()

# %% ----- plotting
# file = tables.open_file("su8_sample_100GHz_coarse.h5", "r")
# file = tables.open_file("_overwrite.h5", "r")
# data = file.root.data
# absorbance = file.root.absorbance

# for i in range(50, 150):
#     fig, ax = plt.subplots(1, 1)
#     ax.hist(absorbance[:, :, i].flatten(), bins=100)
#     ax.set_title(i)

# tau = ppifg * 500 / 1e9
# factor = 1 + 10 * tau
# img = simpson(absorbance[:, :, 80:120], axis=-1)
# y = np.arange(img.shape[0]) * 1.75 / factor
# x = np.arange(img.shape[1]) * 1.75

# %%
# for i in range(10, 35):
#     fig, ax = plt.subplots(1, 1)
#     ax.pcolormesh(x, y, img[::-1, ::-1], cmap='cividis', vmin=i)
#     ax.set_aspect("equal")
#     ax.set_title(i)
#     fig.tight_layout()
