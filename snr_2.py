import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import phase_correction as pc
import scipy.signal as si

cr.style_sheet()

path = r"/home/peterchang/SynologyDrive/Research_Projects/Microscope/FreeRunningSpectra/11-09-2022/"
bckgnd = np.load(path + "stage1_5116_stage2_8500_phase_corrected.npy", mmap_mode='r')
su8 = np.load(path + "stage1_5300_stage2_8970_phase_corrected.npy", mmap_mode='r')
ppifg = len(bckgnd[0])
center = ppifg // 2

apod = 2 ** 14
data = bckgnd
if apod:
    data = data[:, center - apod // 2: center + apod // 2]
else:
    data = data

# high pass filter
lp_nyq = 0.001
b_hp, a_hp = si.butter(4, lp_nyq, 'high')

# calculate the average

# first element of array
avg = data[0]
avg = avg - np.mean(avg)  # subtract mean
avg = si.filtfilt(b_hp, a_hp, avg)  # high pass filter

n = 1
for dat in data[1:]:
    dat = dat - np.mean(dat)  # subtract mean
    dat = si.filtfilt(b_hp, a_hp, dat)  # high pass filter

    avg = avg * n / (n + 1) + dat / (n + 1)  # include in average

    n += 1
    print(len(data) - n)

freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
ft = abs(pc.fft(avg))
plt.plot(freq, ft)
ll = freq.max() * lp_nyq
plt.axvline(ll, color='r')
