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

cr.style_sheet()

plt.ion()

# ______________________________________ load data _____________________________________________________________________
if os.name == 'posix':
    path = r"/Users/peterchang/SynologyDrive/Research_Projects/Microscope/FreeRunningSpectra/11-09-2022/"
else:
    path = r"C:\\Users\\pchan\\SynologyDrive\\Research_Projects\\Microscope\\FreeRunningSpectra\\11-09-2022/"
bckgnd = np.load(path + "stage1_5116_stage2_8500_phase_corrected.npy", mmap_mode='r')
su8 = np.load(path + "stage1_5300_stage2_8970_phase_corrected.npy", mmap_mode='r')
ppifg = len(bckgnd[0])
center = ppifg // 2
frep = 1e9

# ______________________________________ apodization ___________________________________________________________________
apod = 1000
bckgnd = bckgnd[:, center - apod // 2:center + apod // 2]
su8 = su8[:, center - apod // 2:center + apod // 2]


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

    return np.c_[t, NOISE], avg


# _____________________________________ calculate absorbance noise _____________________________________________________
sigma_bckgnd, avg_bckgnd = calculate_snr(bckgnd)
sigma_su8, avg_su8 = calculate_snr(su8)

# _____________________________________ generating figures for CLEO ____________________________________________________
Nyq_freq = frep * center
nu = np.linspace(0, Nyq_freq, center) + Nyq_freq * 2
wl = sc.c / nu * 1e6
dt = ppifg / frep

bckgnd_100 = abs(dpc.fft(np.mean(bckgnd[:int(1e2)], axis=0)))[center:]
su8_100 = abs(dpc.fft(np.mean(su8[:int(1e2)], axis=0)))[center:]
bckgnd_1000 = abs(dpc.fft(np.mean(bckgnd[:int(1e3)], axis=0)))[center:]
su8_1000 = abs(dpc.fft(np.mean(su8[:int(1e3)], axis=0)))[center:]

# ________________ figure iteration 1 __________________________________________________________________________________
fig, ax = plt.subplots(1, 2, figsize=np.array([13.04, 4.8]))
ax[0].plot(wl, -np.log(su8_100 / bckgnd_100),
           # label=f"{np.round(dt * 100 * 1e3, 2)} ms",
           label=f"100 averages"
           )
ax[0].plot(wl, -np.log(su8_1000 / bckgnd_1000),
           # label=f"{np.round(dt * 1000 * 1e3, 2)} ms",
           label=f"1000 averages"
           )
ax[0].plot(wl, -np.log(abs(dpc.fft(avg_su8))[center:] / abs(dpc.fft(avg_bckgnd))[center:]),
           # label=f"{np.round(dt * len(bckgnd), 1)} s",
           label=f"{len(bckgnd)} averages"
           )
ax[0].legend(loc='best')
ax[0].set_xlim(3.25, 3.6)
ax[0].set_ylim(-.5, 2.5)
ax[0].set_ylabel("absorbance")
ax[0].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax[1].loglog(sigma_su8[:, 0], sigma_su8[:, 1], 'o')
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("absorbance noise")
axis = ax[1].twiny()
axis.loglog(sigma_su8[:, 0] / dt, sigma_su8[:, 1], 'o')
axis.set_xlabel("# of averages")

axins = inset_axes(ax[0], width="25%", height="30%", loc="upper right")
axins.plot(wl, abs(dpc.fft(avg_bckgnd))[center:] / 30e3)
axins.plot(wl, abs(dpc.fft(avg_su8))[center:] / 30e3)
axins.set_ylim(0, 1)
axins.set_xlim(2.9, 3.8)
axins.set_yticks([])

# ________________ figure iteration 2 __________________________________________________________________________________
fig, ax = plt.subplots(1, 3, figsize=np.array([14.78, 4.8]))
ax[1].plot(wl, -np.log(su8_100 / bckgnd_100),
           # label=f"{np.round(dt * 100 * 1e3, 2)} ms",
           label=f"100 averages"
           )
ax[1].plot(wl, -np.log(su8_1000 / bckgnd_1000),
           # label=f"{np.round(dt * 1000 * 1e3, 2)} ms",
           label=f"1000 averages"
           )
ax[1].plot(wl, -np.log(abs(dpc.fft(avg_su8))[center:] / abs(dpc.fft(avg_bckgnd))[center:]),
           # label=f"{np.round(dt * len(bckgnd), 1)} s",
           label=f"{len(bckgnd)} averages"
           )
ax[1].legend(loc='best')
ax[1].set_xlim(3.25, 3.6)
ax[1].set_ylim(-.5, 2.5)
ax[1].set_ylabel("absorbance")
ax[1].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax[0].plot(wl, abs(dpc.fft(avg_bckgnd))[center:] / 30e3, label='background')
ax[0].plot(wl, abs(dpc.fft(avg_su8))[center:] / 30e3, label='SU-8')
ax[0].set_ylim(0, 1)
ax[0].set_xlim(2.9, 3.8)
ax[0].set_ylabel("a.u.")
ax[0].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax[0].legend(loc='best')
ax[2].loglog(sigma_su8[:, 0], sigma_su8[:, 1], 'o')
ax[2].set_xlabel("time (s)")
ax[2].set_ylabel("absorbance noise")
axis = ax[2].twiny()
axis.loglog(sigma_su8[:, 0] / dt, sigma_su8[:, 1], 'o')
axis.set_xlabel("# of averages")
