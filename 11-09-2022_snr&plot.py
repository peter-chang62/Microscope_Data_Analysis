"""this is looking at the output of scratch-2.py"""

import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.signal as si
from numpy import ma
from scipy.interpolate import InterpolatedUnivariateSpline

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


def apply_filter(ft, lst_fltrs=list_filter):
    rfreq = np.fft.rfftfreq((len(ft) - 1) * 2, 1e-9) * 1e-6
    for f in lst_fltrs:
        ft = filt(rfreq, *f, ft, "notch")

    return ft


# %% __________________________________________________________________________
def calculate_snr(data, apod=None):
    ppifg = len(data[0])
    center = ppifg // 2

    if apod is not None:
        assert isinstance(int, apod), "apod must be an integer"
        data = data[:, center - apod // 2:center + apod // 2]
    freq = np.fft.rfftfreq(len(data[0]))

    avg = np.mean(data, 0)
    avg -= np.mean(avg)
    ft_avg = np.fft.rfft(avg)

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
        absorbance = -np.log(absorption)
        noise = np.std(absorbance)
        NOISE[n] = noise

        print(len(data) - n - 1)

    return NOISE


# %% __________________________________________________________________________
# data = np.load("data/phase_corrected/"
#                "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#                mmap_mode='r')
#
# noise_1e3 = calculate_snr(data, int(1e3))
# noise_1e4 = calculate_snr(data, int(1e4))
# noise_full = calculate_snr(data, None)
# 
# np.save("data/phase_corrected/bckgnd/sigma_full.npy", noise_full)
# np.save("data/phase_corrected/bckgnd/sigma_1e4.npy", noise_1e4)
# np.save("data/phase_corrected/bckgnd/sigma_1e3.npy", noise_1e3)

# %% __________________________________________________________________________
# data = np.load("data/phase_corrected/"
#                "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
#                mmap_mode='r')
# 
# noise_1e3 = calculate_snr(data, int(1e3))
# noise_1e4 = calculate_snr(data, int(1e4))
# noise_full = calculate_snr(data, None)
# 
# np.save("data/phase_corrected/su8/sigma_full.npy", noise_full)
# np.save("data/phase_corrected/su8/sigma_1e4.npy", noise_1e4)
# np.save("data/phase_corrected/su8/sigma_1e3.npy", noise_1e3)

# %% __________________________________________________________________________
# data_bckgnd = np.load("data/phase_corrected/"
#                       "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#                       mmap_mode='r')
# data_su8 = np.load("data/phase_corrected/"
#                    "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
#                    mmap_mode='r')
# avg_bckgnd = np.mean(data_bckgnd, 0)
# avg_su8 = np.mean(data_su8, 0)

avg_bckgnd = np.load("data/phase_corrected/bckgnd/avg_bckgnd.npy")
avg_su8 = np.load("data/phase_corrected/su8/avg_su8.npy")

ppifg = len(avg_bckgnd)
center = ppifg // 2

ft_bckgnd_full = np.fft.rfft(avg_bckgnd)
ft_bckgnd_1e4 = np.fft.rfft(
    avg_bckgnd[center - int(1e4) // 2:center + int(1e4) // 2])
ft_bckgnd_1e3 = np.fft.rfft(
    avg_bckgnd[center - int(1e3) // 2:center + int(1e3) // 2])

ft_su8_full = np.fft.rfft(avg_su8)
ft_su8_1e4 = np.fft.rfft(
    avg_su8[center - int(1e4) // 2:center + int(1e4) // 2])
ft_su8_1e3 = np.fft.rfft(
    avg_su8[center - int(1e3) // 2:center + int(1e3) // 2])

noise_bckgnd_full = np.load("data/phase_corrected/bckgnd/sigma_full.npy")
noise_bckgnd_1e4 = np.load("data/phase_corrected/bckgnd/sigma_1e4.npy")
noise_bckgnd_1e3 = np.load("data/phase_corrected/bckgnd/sigma_1e3.npy")

noise_su8_full = np.load("data/phase_corrected/su8/sigma_full.npy")
noise_su8_1e4 = np.load("data/phase_corrected/su8/sigma_1e4.npy")
noise_su8_1e3 = np.load("data/phase_corrected/su8/sigma_1e3.npy")

# together
list_filter_plot = list_filter.copy()
list_filter_plot[:, 0] += -.5
list_filter_plot[:, 1] += .5

list_filter_plot_2 = list_filter.copy()
list_filter_plot_2[:, 0] += -3
list_filter_plot_2[:, 1] += 3

plt.figure()
plt.plot(np.fft.rfftfreq((len(ft_bckgnd_full) - 1) * 2, 1e-9) * 1e-6,
         apply_filter(ft_bckgnd_full.__abs__()),
         label='full')
plt.plot(np.fft.rfftfreq((len(ft_bckgnd_1e4) - 1) * 2, 1e-9) * 1e-6,
         apply_filter(ft_bckgnd_1e4.__abs__(), list_filter_plot),
         label='10,000 pts')
plt.plot(np.fft.rfftfreq((len(ft_bckgnd_1e3) - 1) * 2, 1e-9) * 1e-6,
         apply_filter(ft_bckgnd_1e3.__abs__(), list_filter_plot_2),
         label='1,000 pts')
plt.legend(loc='best')
plt.xlabel("MHz")

plt.figure()
plt.plot(np.fft.rfftfreq((len(ft_su8_full) - 1) * 2, 1e-9) * 1e-6,
         apply_filter(ft_su8_full.__abs__()),
         label='full')
plt.plot(np.fft.rfftfreq((len(ft_su8_1e4) - 1) * 2, 1e-9) * 1e-6,
         apply_filter(ft_su8_1e4.__abs__(), list_filter_plot),
         label='10,000 pts')
plt.plot(np.fft.rfftfreq((len(ft_su8_1e3) - 1) * 2, 1e-9) * 1e-6,
         apply_filter(ft_su8_1e3.__abs__(), list_filter_plot_2),
         label='1,000 pts')
plt.legend(loc='best')
plt.xlabel("MHz")

dt = 74180 * 1e-9
t = np.arange(0, len(noise_bckgnd_full) * dt, dt)

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


# %% __________________________________________________________________________
# a little hard to say, the averaging isn't that good
def factor_bckgnd_1e3(t_avg):
    dt = 74180 * 1e-9
    t = np.arange(0, len(noise_bckgnd_full) * dt, dt)

    spl = InterpolatedUnivariateSpline(t, noise_bckgnd_full)
    t_new = InterpolatedUnivariateSpline(t,
                                         noise_bckgnd_1e3 - spl(t_avg)).roots()
    # return t_avg / t_new
    return np.mean(t_avg / t_new)


def factor_bckgnd_1e4(t_avg):
    dt = 74180 * 1e-9
    t = np.arange(0, len(noise_bckgnd_full) * dt, dt)

    spl = InterpolatedUnivariateSpline(t, noise_bckgnd_full)
    t_new = InterpolatedUnivariateSpline(t,
                                         noise_bckgnd_1e4 - spl(t_avg)).roots()
    # return t_avg / t_new
    return np.mean(t_avg / t_new)


def factor_su8_1e3(t_avg):
    dt = 74180 * 1e-9
    t = np.arange(0, len(noise_su8_full) * dt, dt)

    spl = InterpolatedUnivariateSpline(t, noise_su8_full)
    t_new = InterpolatedUnivariateSpline(t,
                                         noise_su8_1e3 - spl(t_avg)).roots()
    # return t_avg / t_new
    return np.mean(t_avg / t_new)


def factor_su8_1e4(t_avg):
    dt = 74180 * 1e-9
    t = np.arange(0, len(noise_su8_full) * dt, dt)

    spl = InterpolatedUnivariateSpline(t, noise_su8_full)
    t_new = InterpolatedUnivariateSpline(t,
                                         noise_su8_1e4 - spl(t_avg)).roots()
    # return t_avg / t_new
    return np.mean(t_avg / t_new)

# t_avg = np.arange(dt, dt * 13480, dt * 10)
# hey = [factor_bckgnd_1e3(i) for i in t_avg]
