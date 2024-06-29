# %% -----
import numpy as np
import matplotlib.pyplot as plt
import pynlo
from scipy.constants import c
from numpy.fft import fftshift, ifftshift, irfft, rfft, rfftfreq
from tqdm import tqdm
import clipboard


# %% -----
path = r"/Volumes/Peter SSD 2/photothermal_v2_1/"
name = "MokuDataLoggerData_20240329_141431.npy"

# %% ----- file conversion from the Moku
# file = path + name
# data = np.load(file, mmap_mode="r")

# data_numpy = np.zeros((data.shape[0], 3))
# for i in tqdm(range(data_numpy.shape[0])):
#     data_numpy[i][0] = data[i][0]
#     data_numpy[i][1] = data[i][1]
#     data_numpy[i][2] = data[i][2]

# save_path = path + "unpacked/"
# file_save = save_path + name
# np.save(file_save, data_numpy)

# %% -----
path += "unpacked/"
file = path + name
data = np.load(file)

dcs = data[:, 2]
ptt = data[:, 1]

# %% -----
level = 0.0111
(idx,) = (dcs > level).nonzero()
spacing = np.diff(idx)
ppifg = int(np.round(spacing[spacing > 150e3].mean()))
ppifg = ppifg if ppifg % 2 == 0 else ppifg + 1
center = ppifg // 2

# idx = idx[:-1][spacing > 150e3]
# N = idx.size
# if idx[0] - center < 0:
#     N -= 1
#     idx = idx[1:]

# dcs_slice = np.zeros((N, ppifg))
# ptt_slice = np.zeros((N, ppifg))
# for i in range(idx.size):
#     dcs_slice[i] = dcs[idx[i] - center : idx[i] + center]
#     ptt_slice[i] = ptt[idx[i] - center : idx[i] + center]

#     roll = center - (dcs_slice[i] ** 2).argmax()
#     dcs_slice[i] = np.roll(dcs_slice[i], roll)
#     ptt_slice[i] = np.roll(ptt_slice[i], roll)

start = dcs[:ppifg].argmax()
start = start + center if start < center else start - center
dcs = dcs[start:]
ptt = ptt[start:]

N = dcs.size // ppifg
dcs = dcs[: N * ppifg]
ptt = ptt[: N * ppifg]
dcs_slice = dcs.reshape((N, ppifg))
ptt_slice = ptt.reshape((N, ppifg))

for n in range(dcs_slice.shape[0]):
    roll = center - (dcs_slice[n] ** 2).argmax()
    dcs_slice[n] = np.roll(dcs_slice[n], roll)
    ptt_slice[n] = np.roll(ptt_slice[n], roll)

# %% -----
resolution = 250
window = ppifg // resolution
window = window if window % 2 == 0 else window + 1
freq_a = rfftfreq(window)
freq = rfftfreq(ppifg)

# x_lims = (0.3392053543261456, 0.3822092550679667)
x_lims = (0.3885124804476703, 0.4455935000120686)
(idx,) = np.logical_and(x_lims[0] < freq_a, freq_a < x_lims[1]).nonzero()

for i in tqdm(range(dcs_slice.shape[0])):
    ft = rfft(ifftshift(dcs_slice[i][center - window // 2 : center + window // 2]))
    p = np.unwrap(np.angle(ft))
    polyfit = np.polyfit(freq_a[idx], p[idx], deg=2)
    poly1d = np.poly1d(polyfit)
    p = poly1d(freq)

    ft = rfft(ifftshift(dcs_slice[i]))
    ft *= np.exp(-1j * p)
    dcs_slice[i] = fftshift(irfft(ft))

    ft = rfft(ifftshift(ptt_slice[i]))
    ft *= np.exp(-1j * p)
    ptt_slice[i] = fftshift(irfft(ft))

    if i % 10 == 0:
        plt.plot(dcs_slice[i][center - 100 : center + 100])
        plt.pause(0.01)

avg_dcs = np.mean(dcs_slice, axis=0)
avg_ptt = np.mean(ptt_slice, axis=0)

# %% -----
avg_ptt = np.roll(avg_ptt, center - (avg_ptt**2).argmax())

# %% -----
resolution = 300
window = ppifg // resolution
window = window if window % 2 == 0 else window + 1
freq = rfftfreq(window)

s_dcs = abs(rfft(ifftshift(avg_dcs[center - window // 2 : center + window // 2])))
s_ptt = abs(rfft(ifftshift(avg_ptt[center - window // 2 : center + window // 2])))

fig, ax = plt.subplots(1, 1)
ax.plot(freq, s_dcs, label="dcs")
ax.plot(freq, s_ptt, label="photothermal dcs")
ax.legend(loc="best")
ax.set_xlabel("fraction of Nyquist")
fig.tight_layout()

# %% -----
# save_path = path + "averaged/"
# file = save_path + name
# np.save(file, np.c_[data[:, 0][:ppifg], avg_dcs, avg_ptt])
