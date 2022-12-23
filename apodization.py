import sys
sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.signal as si
import os
import scipy.constants as sc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import digital_phase_correction as dpc
import td_phase_correct as td
cr.style_sheet()
plt.ion()


def calculate_snr(data, plot=False):
    # ______________________________________ calculate background ______________________________________________________
    avg = np.mean(data, 0)
    avg = avg - np.mean(avg)
    ft_avg = dpc.fft(avg).__abs__()

    # low pass filtered background
    b_lp, a_lp = si.butter(4, .20, 'low')
    filtered = si.filtfilt(b_lp, a_lp, ft_avg)

    # area to calculate snr from
    freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
    ll = np.argmin(abs(freq - .10784578053383662))
    ul = np.argmin(abs(freq - .19547047721757888))
    denom = filtered[ll:ul].copy()

    # ______________________________________ calculate cumulative average ______________________________________________
    x = data[0]
    x = x - np.mean(x)
    ft = dpc.fft(x).__abs__()
    num = ft[ll:ul]
    absorption = num / denom
    absorbance = - np.log(absorption)
    noise = np.std(absorbance)

    NOISE = np.zeros(len(data))
    NOISE[0] = noise

    n = 1
    for dat in data[1:]:
        x = x * n / (n + 1) + dat / (n + 1)
        x = x - np.mean(x)
        ft = dpc.fft(x).__abs__()
        num = ft[ll:ul]
        absorption = num / denom
        absorbance = - np.log(absorption)
        noise = np.std(absorbance)
        NOISE[n] = noise

        n += 1
        print(len(data) - 1 - n)

    dt = ppifg / frep
    t = np.arange(0, len(NOISE) * dt, dt)
    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.loglog(t, NOISE, 'o')
        ax.set_xlabel("time (s)")
        ax.set_ylabel("$\\mathrm{\\sigma}$")

    return np.c_[t, NOISE], avg, filtered


# ______________________________________ load data _____________________________________________________________________
if os.name == 'posix':
    path = r"/Users/peterchang/SynologyDrive/Research_Projects/Microscope/FreeRunningSpectra/11-09-2022/"
else:
    path = r"C:\\Users\\pchan\\SynologyDrive\\Research_Projects\\Microscope\\FreeRunningSpectra\\11-09-2022/"

read_mode = "r"
assert read_mode == "r", "if not you will literally overwrite the data file!!"
bckgnd = np.load(path + "stage1_5116_stage2_8500_phase_corrected.npy", mmap_mode=read_mode)
su8 = np.load(path + "stage1_5300_stage2_8970_phase_corrected.npy", mmap_mode=read_mode)

ppifg = len(bckgnd[0])
center = ppifg // 2
frep = 1e9

# ______________________________________ apodize and re-phase correct __________________________________________________
# apodize
apod = 1000
bckgnd = bckgnd[:, center - apod // 2: center + apod // 2]
su8 = su8[:, center - apod // 2: center + apod // 2]

# phase correct
N = len(su8)
zoom = 30
X = np.zeros((N, apod))
for n, x in enumerate(su8[:N]):
    opt = td.Optimize(np.vstack([su8[0], x]))
    opt.phase_correct(zoom=zoom)
    corr = opt.CORR[1]
    X[n] = corr
    print(len(su8[:N]) - n - 1)

# calculate SNR of apodized + phase corrected data
sigma, avg, filt_avg = calculate_snr(X)

# ______________________________________ plotting ______________________________________________________________________
plt.figure()
plt.loglog(sigma[:, 0], sigma[:, 1], '.')

plt.figure()
[plt.plot(i) for i in X[::100]]
plt.plot(avg, 'k')
