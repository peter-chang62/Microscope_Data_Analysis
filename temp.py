import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
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
    ft = [filt(rfreq, *i, ft, "notch") for i in list_filter]
    return ft


# %% __________________________________________________________________________
full = np.load("data/phase_corrected/bckgnd/full.npy")
_1e4 = np.load("data/phase_corrected/bckgnd/1e4_pts_sigma.npy")
_1e3 = np.load("data/phase_corrected/bckgnd/1e3_pts_sigma.npy")

# fig 2
plt.figure()
plt.loglog(full[:, 0], full[:, 1], 'o',
           label="full interferogram")
plt.plot(_1e4[:, 0], _1e4[:, 1], 'o', label="10,000 pt window")
plt.plot(_1e3[:, 0], _1e3[:, 1], 'o', label="1000 pt window")
plt.xlabel("time (s)")
plt.ylabel("absorbance noise (a.u.)")
plt.legend(loc='best')

# %% __________________________________________________________________________
full = np.load("data/phase_corrected/su8/full.npy")
_1e4 = np.load("data/phase_corrected/su8/1e4_pts_sigma.npy")
_1e3 = np.load("data/phase_corrected/su8/1e3_pts_sigma.npy")

# fig 3
plt.figure()
plt.loglog(full[:, 0], full[:, 1], 'o',
           label="full interferogram")
plt.plot(_1e4[:, 0], _1e4[:, 1], 'o', label="10,000 pt window")
plt.plot(_1e3[:, 0], _1e3[:, 1], 'o', label="1000 pt window")
plt.xlabel("time (s)")
plt.ylabel("absorbance noise (a.u.)")
plt.legend(loc='best')

# %% __________________________________________________________________________
# avg = np.load("data/phase_corrected/"
#               "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#               mmap_mode='r')
# x = 0
# for n, i in enumerate(avg):
#     ft = np.fft.rfft(i)
#     x = (x * n + apply_filter(ft)) / (n + 1)
#     print(len(avg) - n - 1)
# np.savez_compressed("data/phase_corrected/bckgnd/ft_avg.npz", data=x.data,
#                     mask=x.mask)

ft_avg = np.load("data/phase_corrected/bckgnd/ft_avg.npz")
ft_avg = ma.array(**ft_avg)
avg = np.fft.irfft(ft_avg)

ppifg = len(avg)
center = ppifg // 2

_1e4_avg = avg[center - int(1e4) // 2: center + int(1e4) // 2]
_1e3_avg = avg[center - int(1e3) // 2: center + int(1e3) // 2]

# fig 4
plt.figure()
plt.plot(np.fft.rfftfreq(len(avg)),
         np.fft.rfft(avg).__abs__(), label="full interferogram")
plt.plot(np.fft.rfftfreq(len(_1e4_avg)),
         np.fft.rfft(_1e4_avg).__abs__(), label="10,000 pt window")
plt.plot(np.fft.rfftfreq(len(_1e3_avg)),
         np.fft.rfft(_1e3_avg).__abs__(), label="1000 pt window")
plt.legend(loc='best')
plt.xlabel("MHz")
plt.axvline(.10784578053383662, color='k', linestyle='--')
plt.axvline(.19547047721757888, color='k', linestyle='--')

# %% __________________________________________________________________________
# avg = np.load("data/phase_corrected/"
#               "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
#               mmap_mode='r')
# x = 0
# for n, i in enumerate(avg):
#     ft = np.fft.rfft(i)
#     x = (x * n + apply_filter(ft)) / (n + 1)
#     print(len(avg) - n - 1)
# np.savez_compressed("data/phase_corrected/su8/ft_avg.npz", data=x.data,
#                     mask=x.mask)

ft_avg = np.load("data/phase_corrected/su8/ft_avg.npz")
ft_avg = ma.array(**ft_avg)
avg = np.fft.irfft(ft_avg)

_1e4_avg = avg[center - int(1e4) // 2: center + int(1e4) // 2]
_1e3_avg = avg[center - int(1e3) // 2: center + int(1e3) // 2]

# fig 5
plt.figure()
plt.plot(np.fft.rfftfreq(len(avg)),
         np.fft.rfft(avg).__abs__(), label="full interferogram")
plt.plot(np.fft.rfftfreq(len(_1e4_avg)),
         np.fft.rfft(_1e4_avg).__abs__(), label="10,000 pt window")
plt.plot(np.fft.rfftfreq(len(_1e3_avg)),
         np.fft.rfft(_1e3_avg).__abs__(), label="1000 pt window")
plt.legend(loc='best')
plt.xlabel("MHz")
plt.axvline(.10784578053383662, color='k', linestyle='--')
plt.axvline(.19547047721757888, color='k', linestyle='--')

plt.show()
