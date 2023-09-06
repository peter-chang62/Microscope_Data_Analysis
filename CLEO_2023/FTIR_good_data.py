# shape goes as bands, lines, samples

# %% -----
import numpy as np
import matplotlib.pyplot as plt
import clipboard as cr
from scipy.integrate import simpson
from tqdm import tqdm
import os
import platform

data_index = 5
wnum = np.genfromtxt("wnum.csv", delimiter=",")

if platform.system().lower() == "darwin":
    path = r"/Users/peterchang/Resilio Sync/July ovarian FTIR I3/good data/"
elif platform.system().lower() == "windows":
    path = r"C:\\Users\\pchan\\Data\\July ovarian FTIR I3\\good data/"
folder = [
    "wp_ov-63_hd_16ca/",
    "wp_ov-63_sd_1ca/",
    "wp_ov-63_sd_16ca/",
    "wp_ov-65_hd_1ca/",
    "wp_ov-65_sd_1ca/",
    "wp_ov-65_sd_16ca/",
]
bands = 394
samples = [
    2000,
    300,
    280,
    1430,
    270,
    300,
]
lines = [
    1500,
    300,
    280,
    1430,
    270,
    300,
]

x = np.memmap(path + folder[data_index] + "I3_Cropped", dtype="<f")
x.resize(bands, lines[data_index], samples[data_index])

x_i = np.zeros((x.shape[1], x.shape[2]), dtype=np.float32)
for i in tqdm(range(x.shape[1])):
    x_i[i] = simpson(x[:, i, :], axis=0)

# %% -----
(idx_b,) = np.logical_and(2900 < wnum, wnum < 2960).nonzero()  # big C-H peak
(idx_s,) = np.logical_and(2838 < wnum, wnum < 2872).nonzero()  # small C-H peak

x_i_peak_b = np.zeros((x.shape[1], x.shape[2]), dtype=np.float32)
for i in tqdm(range(x.shape[1])):
    x_i_peak_b[i] = x[idx_b][:, i, :].max(axis=0)

x_i_peak_s = np.zeros((x.shape[1], x.shape[2]), dtype=np.float32)
for i in tqdm(range(x.shape[1])):
    x_i_peak_s[i] = x[idx_s][:, i, :].max(axis=0)

# %% -----
fig, ax = plt.subplots(1, 1, num=f"file {data_index}: integration over all wavelengths")
ax.imshow(x_i, vmin=9, vmax=120)
fig.tight_layout()

fig, ax = plt.subplots(1, 1, num=f"file {data_index}: big C-H peak")
ax.imshow(x_i_peak_b)
fig.tight_layout()

fig, ax = plt.subplots(1, 1, num=f"file {data_index}: small C-H peak")
ax.imshow(x_i_peak_s)
fig.tight_layout()

# %% ----- save image data
os.mkdir("I3_good_data/" + folder[data_index])
np.save(f"I3_good_data/{folder[data_index]}img_integrated_absorbance.npy", x_i)
np.save(f"I3_good_data/{folder[data_index]}img_big_CH_peak.npy", x_i_peak_b)
np.save(f"I3_good_data/{folder[data_index]}img_small_CH_peak.npy", x_i_peak_s)
