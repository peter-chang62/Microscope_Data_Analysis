import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import phase_correction as pc
import scipy.signal as si

cr.style_sheet()

# ______________________________________ load data _____________________________________________________________________
path = r"C:\Users\pchan\SynologyDrive\Research_Projects\Microscope\FreeRunningSpectra\11-09-2022/"
bckgnd = np.load(path + "stage1_5116_stage2_8500_phase_corrected.npy", mmap_mode='r')
su8 = np.load(path + "stage1_5300_stage2_8970_phase_corrected.npy", mmap_mode='r')
ppifg = len(bckgnd[0])
center = ppifg // 2

# ______________________________________ calculate background __________________________________________________________
data = bckgnd
avg = np.mean(data, 0)
avg = avg - np.mean(avg)
ft_avg = pc.fft(avg).__abs__()

# low pass filtered background
b_lp, a_lp = si.butter(4, .20, 'low')
filtered = si.filtfilt(b_lp, a_lp, ft_avg)

# area to calculate snr from
freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
ll = np.argmin(abs(freq - .10784578053383662))
ul = np.argmin(abs(freq - .19547047721757888))
denom = filtered[ll:ul].copy()

# ______________________________________ calculate cumulative average __________________________________________________
x = data[0]
x = x - np.mean(x)
ft = pc.fft(x).__abs__()
num = ft[ll:ul]
noise = np.std(num / denom)

NOISE = np.zeros(len(data))
NOISE[0] = noise

n = 1
for dat in data[1:]:
    x = x * n / (n + 1) + dat / (n + 1)
    x = x - np.mean(x)
    ft = pc.fft(x).__abs__()
    num = ft[ll:ul]
    noise = np.std(num / denom)
    NOISE[n] = noise

    n += 1
    print(len(data) - 1 - n)

dt = 1 / 13.47e3
t = np.arange(0, len(NOISE) * dt, dt)
fig, ax = plt.subplots(1, 1)
ax.loglog(t, NOISE, 'o')
ax.set_xlabel("time (s)")
ax.set_ylabel("$\mathrm{\sigma}$")
