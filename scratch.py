import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import os

cr.style_sheet()


def rfft(x):
    return np.fft.rfft(x)


def irfft(x):
    return np.fft.irfft(x)


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

data = np.load(path + "stage1_5116_stage2_8500_53856x74180.npy",
               mmap_mode='r')

ppifg = len(data[0])
center = ppifg // 2
N_IFG = len(data)

data.resize((ppifg * N_IFG,))
data = data[center:-center]
data.resize((N_IFG - 1, ppifg))
data = data[::50]
freq = np.fft.ifftshift(np.fft.fftfreq(len(data[0]), 1e-9) * 1e-6)

# %% __________________________________________________________________________
# straight up regular phase correction (time domain)
opt = td.Optimize(data[:, center - 10:center + 10])
corr = data.copy()
opt.phase_correct(corr)

# %% __________________________________________________________________________
# phase correct interferograms to overlap the f0 oscillations
# subst = data[:, :1000]
# opt = td.Optimize(subst)
# corr = data.copy()
# opt.phase_correct(corr)
#
# # %% look at the difference for the averages (I know it's wrong because
# # you didn't overlap the centerbursts), for when the centerburst is removed
# # and when it's left in. It looks weird
# avg = np.mean(corr, 0)
# bckgnd = np.hstack([avg[:center - 5], avg[center + 5:]])
# plt.plot(np.fft.fftfreq(len(bckgnd), 1e-9) * 1e-6,
#          dpc.fft(bckgnd).__abs__())
# plt.plot(freq, dpc.fft(avg).__abs__())
# plt.ylim(0, 1.2e4)


# %% __________________________________________________________________________
# x = data[0].copy()
# ft = rfft(x)
# rfreq = np.fft.rfftfreq(len(x), 1e-9) * 1e-6
# w_filt = filt(rfreq, 270, 295, ft, "notch")
# t_filt = np.fft.irfft(w_filt)
#
# # %%
# plt.figure()
# plt.plot(freq, dpc.fft(x).__abs__())
# plt.plot(rfreq, ft.__abs__())
# plt.plot(rfreq, w_filt.__abs__())
#
# # %%
# plt.figure()
# plt.plot(x)
# plt.plot(t_filt)

# %% __________________________________________________________________________
# filter the data first (using the 1 GHz resolution)
data_filt = np.asarray(data).copy()  # temporary solution (looking at a subset)
rfreq = np.fft.rfftfreq(len(data_filt[0]), 1e-9) * 1e-6
for n, x in enumerate(data_filt):
    ft = rfft(x)
    w_filt = filt(rfreq, 0, 1, ft, "notch")
    w_filt = filt(rfreq, 31.5, 32.5, w_filt, "notch")
    w_filt = filt(rfreq, 249.5, 250.5, w_filt, "notch")
    w_filt = filt(rfreq, 281, 283, w_filt, "notch")
    w_filt = filt(rfreq, 336, 339.5, w_filt, "notch")
    w_filt = filt(rfreq, 499, 500, w_filt, "notch")

    t_filt = np.fft.irfft(w_filt)
    data_filt[n] = t_filt
    print(len(data_filt) - n - 1)

# %%
opt = td.Optimize(data_filt[:, center - 10:center + 10])
opt.phase_correct(data_filt)

avg = np.mean(data_filt, 0)
ft_avg = np.fft.rfft(avg)
ft_apd = np.fft.rfft(avg[center - 1000:center + 1000])
rfreq_apd = np.fft.rfftfreq(len(avg[center - 1000:center + 1000]), 1e-9) * 1e-6

# %%
fig, ax = plt.subplots(1, 2, figsize=np.array([11.19, 4.8]))
ax[0].plot(rfreq, ft_avg.__abs__())
ax[0].plot(rfreq_apd, ft_apd.__abs__(), label="apodized")
ax[0].legend(loc='best')
ax[0].set_ylim(0, 2.9e4)
ax[0].set_title("unfiltered")

ax[1].plot(rfreq, np.fft.rfft(np.mean(corr, 0)).__abs__())
ax[1].plot(
    np.fft.rfftfreq(len(corr[0][center - 1000:center + 1000]), 1e-9) * 1e-6,
    np.fft.rfft(np.mean(corr[:, center - 1000:center + 1000], 0)).__abs__(),
    label="apodized")
ax[1].legend(loc='best')
ax[1].set_ylim(0, 2.9e4)
ax[1].set_title("filtered")

plt.show()
