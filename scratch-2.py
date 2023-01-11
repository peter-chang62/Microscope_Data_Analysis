"""this is looking at the output of scratch-2.py"""

import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.signal as si
from numpy import ma

cr.style_sheet()

# together
list_filter = np.array([[0, 2],
                        [31, 32.8],
                        [249.5, 250.5],
                        [280, 284],
                        [337, 338],
                        [436, 437],
                        [499, 500]])


def filt(freq, ll, ul, x, type="bp"):
    if type == "bp":
        mask = np.where(np.logical_and(freq > ll, freq < ul), 0, 1)
    elif type == "notch":
        mask = np.where(np.logical_or(freq < ll, freq > ul), 0, 1)

    return ma.array(x, mask=mask)


def apply_filter(ft):
    rfreq = np.fft.rfftfreq((len(ft) - 1) * 2, 1e-9) * 1e-6
    for f in list_filter:
        ft = filt(rfreq, *f, ft, "notch")

    return ft


# %% __________________________________________________________________________
# data = np.load("data/phase_corrected/"
#                "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#                mmap_mode='r')
# ppifg = len(data[0])
# center = ppifg // 2
# apod = int(1e3)
# data = data[:, center - apod // 2:center + apod // 2]
#
# x = 0
# for n, i in enumerate(data):
#     ft = np.fft.rfft(i)
#     x = (x * n + apply_filter(ft)) / (n + 1)
#     print(len(data) - n - 1)
# np.savez_compressed("data/phase_corrected/bckgnd/ft_avg_1e3pts.npz",
#                     data=x.data,
#                     mask=x.mask)

# %% __________________________________________________________________________
# data = np.load("data/phase_corrected/"
#                "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
#                mmap_mode='r')
# ppifg = len(data[0])
# center = ppifg // 2
# apod = int(1e4)
# data = data[:, center - apod // 2:center + apod // 2]
#
# x = 0
# for n, i in enumerate(data):
#     ft = np.fft.rfft(i)
#     x = (x * n + apply_filter(ft)) / (n + 1)
#     print(len(data) - n - 1)
# np.savez_compressed("data/phase_corrected/su8/ft_avg_1e4pts.npz",
#                     data=x.data,
#                     mask=x.mask)


# %% __________________________________________________________________________
def calculate_snr(data, ft_avg, apod=None):
    ppifg = len(data[0])
    center = ppifg // 2

    if apod is not None:
        data = data[:, center - apod // 2:center + apod // 2]
    freq = np.fft.rfftfreq(len(data[0]))
    assert len(freq) == len(ft_avg)

    ll = np.argmin(abs(freq - .10784578053383662))
    ul = np.argmin(abs(freq - .19547047721757888))

    b, a = si.butter(4, .2, "low")
    ft_avg_filt = si.filtfilt(b, a, ft_avg.__abs__())
    denom = ft_avg_filt[ll:ul]

    x = 0
    NOISE = np.zeros(len(data))
    for n, i in enumerate(data):
        i = i - np.mean(i)

        ft = np.fft.rfft(i)
        x = (x * n + apply_filter(ft)) / (n + 1)

        num = x.__abs__()[ll:ul]
        absorption = num / denom
        absorbance = -np.log10(absorption)
        noise = np.std(absorbance)
        NOISE[n] = noise

        print(len(data) - n - 1)

    return NOISE


# %% __________________________________________________________________________
# data = np.load("data/phase_corrected/"
#                "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#                mmap_mode='r')
#
# avg = np.load("data/phase_corrected/bckgnd/ft_avg_1e3pts.npz")
# avg = ma.array(**avg)
# noise_1e3 = calculate_snr(data, avg, int(1e3))
#
# avg = np.load("data/phase_corrected/bckgnd/ft_avg_1e4pts.npz")
# avg = ma.array(**avg)
# noise_1e4 = calculate_snr(data, avg, int(1e4))
#
# avg = np.load("data/phase_corrected/bckgnd/ft_avg_full.npz")
# avg = ma.array(**avg)
# noise_full = calculate_snr(data, avg, None)
#
# np.save("data/phase_corrected/bckgnd/sigma_full.npy", noise_full)
# np.save("data/phase_corrected/bckgnd/sigma_1e4.npy", noise_1e4)
# np.save("data/phase_corrected/bckgnd/sigma_1e3.npy", noise_1e3)

# %% __________________________________________________________________________
# data = np.load("data/phase_corrected/"
#                "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
#                mmap_mode='r')
#
# avg = np.load("data/phase_corrected/su8/ft_avg_1e3pts.npz")
# avg = ma.array(**avg)
# noise_1e3 = calculate_snr(data, avg, int(1e3))
#
# avg = np.load("data/phase_corrected/su8/ft_avg_1e4pts.npz")
# avg = ma.array(**avg)
# noise_1e4 = calculate_snr(data, avg, int(1e4))
#
# avg = np.load("data/phase_corrected/su8/ft_avg_full.npz")
# avg = ma.array(**avg)
# noise_full = calculate_snr(data, avg, None)
#
# np.save("data/phase_corrected/su8/sigma_full.npy", noise_full)
# np.save("data/phase_corrected/su8/sigma_1e4.npy", noise_1e4)
# np.save("data/phase_corrected/su8/sigma_1e3.npy", noise_1e3)

# %% __________________________________________________________________________
ft_bckgnd_full = ma.array(
    **np.load("data/phase_corrected/bckgnd/ft_avg_full.npz"))
ft_bckgnd_1e4 = ma.array(
    **np.load("data/phase_corrected/bckgnd/ft_avg_1e4pts.npz"))
ft_bckgnd_1e3 = ma.array(
    **np.load("data/phase_corrected/bckgnd/ft_avg_1e3pts.npz"))

ft_su8_full = ma.array(**np.load("data/phase_corrected/su8/ft_avg_full.npz"))
ft_su8_1e4 = ma.array(**np.load("data/phase_corrected/su8/ft_avg_1e4pts.npz"))
ft_su8_1e3 = ma.array(**np.load("data/phase_corrected/su8/ft_avg_1e3pts.npz"))

noise_bckgnd_full = np.load("data/phase_corrected/bckgnd/sigma_full.npy")
noise_bckgnd_1e4 = np.load("data/phase_corrected/bckgnd/sigma_1e4.npy")
noise_bckgnd_1e3 = np.load("data/phase_corrected/bckgnd/sigma_1e3.npy")

noise_su8_full = np.load("data/phase_corrected/su8/sigma_full.npy")
noise_su8_1e4 = np.load("data/phase_corrected/su8/sigma_1e4.npy")
noise_su8_1e3 = np.load("data/phase_corrected/su8/sigma_1e3.npy")

dt = 74180 * 1e-9
t = np.arange(0, len(noise_bckgnd_full) * dt, dt)

plt.figure()
plt.plot(np.fft.rfftfreq((len(ft_bckgnd_full) - 1) * 2, 1e-9) * 1e-6,
         ft_bckgnd_full.__abs__(), label='full')
plt.plot(np.fft.rfftfreq((len(ft_bckgnd_1e4) - 1) * 2, 1e-9) * 1e-6,
         ft_bckgnd_1e4.__abs__(), label='10,000 pts')
plt.plot(np.fft.rfftfreq((len(ft_bckgnd_1e3) - 1) * 2, 1e-9) * 1e-6,
         ft_bckgnd_1e3.__abs__(), label='1,000 pts')
plt.legend(loc='best')
plt.xlabel("MHz")

plt.figure()
plt.plot(np.fft.rfftfreq((len(ft_su8_full) - 1) * 2, 1e-9) * 1e-6,
         ft_su8_full.__abs__(), label='full')
plt.plot(np.fft.rfftfreq((len(ft_su8_1e4) - 1) * 2, 1e-9) * 1e-6,
         ft_su8_1e4.__abs__(), label='10,000 pts')
plt.plot(np.fft.rfftfreq((len(ft_su8_1e3) - 1) * 2, 1e-9) * 1e-6,
         ft_su8_1e3.__abs__(), label='1,000 pts')
plt.legend(loc='best')
plt.xlabel("MHz")

plt.figure()
plt.loglog(t, noise_bckgnd_full, 'o', label="full")
plt.loglog(t, noise_bckgnd_1e4, 'o', label="10,000 pts")
plt.loglog(t, noise_bckgnd_1e3, 'o', label="1,000 pts")
plt.xlabel("t (s)")
plt.ylabel("absorbance noise")

plt.figure()
plt.loglog(t, noise_su8_full, 'o', label="full")
plt.loglog(t, noise_su8_1e4, 'o', label="10,000 pts")
plt.loglog(t, noise_su8_1e3, 'o', label="1,000 pts")
plt.xlabel("t (s)")
plt.ylabel("absorbance noise")
