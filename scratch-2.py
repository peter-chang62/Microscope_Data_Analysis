"this is looking at the output of scratch-2.py"

import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.signal as si

cr.style_sheet()

data = np.load("data/phase_corrected/"
               "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
               mmap_mode='r')


def calculate_snr(data, apod=None, plot=False):
    ppifg = len(data[0])
    center = ppifg // 2
    if apod is not None:
        data = data[:, center - apod // 2:center + apod // 2]

    avg = np.mean(data, 0)
    avg -= np.mean(avg)

    b, a = si.butter(4, .20, 'low')
    filt = si.filtfilt(b, a, np.fft.rfft(avg).__abs__())
    freq = np.fft.rfftfreq(len(avg))
    ll = np.argmin(abs(freq - .10784578053383662))
    ul = np.argmin(abs(freq - .19547047721757888))
    denom = filt[ll:ul]

    x = data[0]
    x = x - np.mean(x)
    ft = np.fft.rfft(x).__abs__()
    num = ft[ll:ul]
    absorption = num / denom
    absorbance = -np.log(absorption)
    noise = np.std(absorbance)

    NOISE = np.zeros(len(data))
    NOISE[0] = noise

    n = 1
    for dat in data[1:]:
        x = (x * n + dat) / (n + 1)
        x -= np.mean(x)
        ft = np.fft.rfft(x).__abs__()
        num = ft[ll:ul]
        absorption = num / denom
        absorbance = -np.log(absorption)
        noise = np.std(absorbance)
        NOISE[n] = noise
        n += 1
        print(len(data) - 1 - n)

    dt = ppifg / 1e9
    t = np.arange(0, len(NOISE) * dt, dt)

    if plot:
        plt.figure()
        plt.loglog(t, NOISE, 'o')
        plt.xlabel("time (s)")
        plt.ylabel("$\\mathrm{\\sigma}$")

    class result:
        def __init__(self):
            self.sigma = np.c_[t, NOISE]
            self.avg = avg

    return result()


full = calculate_snr(data, apod=None, plot=False)
_1e4 = calculate_snr(data, apod=int(1e4), plot=False)
_1e3 = calculate_snr(data, apod=int(1e3), plot=False)

# %% save
np.save("data/phase_corrected/full.npy", full.sigma)
np.save("data/phase_corrected/1e4_pts.npy", _1e4.sigma)
np.save("data/phase_corrected/1e3.npy", _1e3.sigma)

# %% plot
# plt.figure()
# plt.loglog(full.sigma[:, 0], full.sigma[:, 1], 'o', label="full interferogram")
# plt.plot(_1e4.sigma[:, 0], _1e4.sigma[:, 1], 'o', label="10,000 pt window")
# plt.plot(_1e3.sigma[:, 0], _1e3.sigma[:, 1], 'o', label="1000 pt window")
# plt.xlabel("time (s)")
# plt.ylabel("absorbance noise (a.u.)")
# plt.legend(loc='best')
#
# plt.figure()
# plt.plot(np.fft.rfftfreq(len(full.avg)),
#          np.fft.rfft(full.avg).__abs__(), label="full interferogram")
# plt.plot(np.fft.rfftfreq(len(_1e4.avg)),
#          np.fft.rfft(_1e4.avg).__abs__(), label="10,000 pt window")
# plt.plot(np.fft.rfftfreq(len(_1e3.avg)),
#          np.fft.rfft(_1e3.avg).__abs__(), label="1000 pt window")
# plt.legend(loc='best')
# plt.xlabel("MHz")
