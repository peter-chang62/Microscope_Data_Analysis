"""It is not as of yet working for stage1_5300_stage2_8970_53856x74180.npy,
because the phase correction misses on some of them (probably bad minima in
the optimization)"""
import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import os

cr.style_sheet()


def filt(freq, ll, ul, x, type="bp"):
    if type == "bp":
        return np.where(np.logical_and(freq > ll, freq < ul), x, 0)

    elif type == "notch":
        return np.where(np.logical_or(freq < ll, freq > ul), x, 0)


if os.name == 'posix':
    path = "/Users/peterchang/SynologyDrive/Research_Projects/" \
           "Microscope/FreeRunningSpectra/11-09-2022/"
else:
    path = r"C:/Users/pchan/SynologyDrive/Research_Projects/Microscope" \
           r"/FreeRunningSpectra/11-09-2022/"

data = np.load(
    path + "stage1_5116_stage2_8500_53856x74180.npy",
    mmap_mode='r')

ppifg = len(data[0])
center = ppifg // 2
N_IFG = len(data)

data.resize((ppifg * N_IFG,))
data = data[center:-center]
data.resize((N_IFG - 1, ppifg))
# lab frame frequency (0 to 500 MHz)
freq = np.fft.ifftshift(np.fft.fftfreq(len(data[0]), 1e-9) * 1e-6)
rfreq = np.fft.rfftfreq(len(data[0]), 1e-9) * 1e-6

# %% __________________________________________________________________________
data = data[:int(1e4)].copy()

h = 0
for n in range(len(data)):
    ft = np.fft.rfft(data[n])  # make sure to load from data
    w_filt = filt(rfreq, 0, 1, ft, "notch")
    w_filt = filt(rfreq, 31.5, 32.5, w_filt, "notch")
    w_filt = filt(rfreq, 249.5, 250.5, w_filt, "notch")
    w_filt = filt(rfreq, 281, 283, w_filt, "notch")
    w_filt = filt(rfreq, 336, 339.5, w_filt, "notch")
    w_filt = filt(rfreq, 499, 500, w_filt, "notch")

    t_filt = np.fft.irfft(w_filt)
    data[n] = t_filt
    print(len(data) - h - 1)
    h += 1

# %% __________________________________________________________________________
# now phase correct the filtered data (1 GHz resolution)
opt = td.Optimize(data[:, center - 50:center + 50])
opt.phase_correct(data,
                  start_index=0,
                  end_index=None,
                  method='Powell')

avg = np.mean(data, 0)
np.save("avg.npy", avg)
