import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import phase_correction as pc
import scipy.signal as si
import os
import scipy.constants as sc

cr.style_sheet()

# ______________________________________ load data _____________________________________________________________________
if os.name == 'posix':
    path = r"/home/peterchang/SynologyDrive/Research_Projects/Microscope/FreeRunningSpectra/11-09-2022/"
else:
    path = r"C:\Users\fastdaq\SynologyDrive\Research_Projects\Microscope\FreeRunningSpectra\11-09-2022/"
bckgnd = np.load(path + "stage1_5116_stage2_8500_phase_corrected.npy", mmap_mode='r')
su8 = np.load(path + "stage1_5300_stage2_8970_phase_corrected.npy", mmap_mode='r')
ppifg = len(bckgnd[0])
center = ppifg // 2
frep = 1e9


def calculate_snr(data, plot=False):
    # ______________________________________ calculate background ______________________________________________________
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

    # ______________________________________ calculate cumulative average ______________________________________________
    x = data[0]
    x = x - np.mean(x)
    ft = pc.fft(x).__abs__()
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
        ft = pc.fft(x).__abs__()
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
        ax.set_ylabel("$\mathrm{\sigma}$")

    return np.c_[t, NOISE], avg


# _____________________________________ calculate absorbance noise _____________________________________________________
sigma_bckgnd, avg_bckgnd = calculate_snr(bckgnd)
sigma_su8, avg_su8 = calculate_snr(su8)

# _____________________________________ generating figures for CLEO ____________________________________________________
Nyq_freq = frep * center
nu = np.linspace(0, Nyq_freq, center) + Nyq_freq * 2
wl = sc.c / nu * 1e6
dt = ppifg / frep

bckgnd_100 = abs(pc.fft(np.mean(bckgnd[:int(1e2)], axis=0)))[center:]
su8_100 = abs(pc.fft(np.mean(su8[:int(1e2)], axis=0)))[center:]
bckgnd_1000 = abs(pc.fft(np.mean(bckgnd[:int(1e3)], axis=0)))[center:]
su8_1000 = abs(pc.fft(np.mean(su8[:int(1e3)], axis=0)))[center:]

fig, ax = plt.subplots(1, 2, figsize=np.array([13.04, 4.8]))
ax[0].plot(wl, -np.log(su8_100 / bckgnd_100), label=f"{np.round(dt * 100 * 1e3, 2)} ms")
ax[0].plot(wl, -np.log(su8_1000 / bckgnd_1000), label=f"{np.round(dt * 1000 * 1e3, 2)} ms")
ax[0].plot(wl, -np.log(abs(pc.fft(avg_su8))[center:] / abs(pc.fft(avg_bckgnd))[center:]),
           label=f"{np.round(dt * len(bckgnd), 1)} s")
ax[0].legend(loc='best')
ax[0].set_xlim(3.25, 3.6)
ax[0].set_ylim(-.5, 2.5)
ax[0].set_ylabel("absorbance")
ax[0].set_xlabel("wavelength $\mathrm{\mu m}$")
ax[1].loglog(sigma_su8[:, 0], sigma_su8[:, 1], 'o')
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("absorbance noise")
