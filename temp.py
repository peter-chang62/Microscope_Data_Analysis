import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr

cr.style_sheet()

x1 = np.load("data/phase_corrected/misc/avg.npy")
x2 = np.load("data/phase_corrected/misc/avg_filt.npy")

ppifg = len(x1)
center = ppifg // 2
a_x1 = x1[center - 1000:center + 1000]
a_x2 = x2[center - 1000:center + 1000]

ft_a_x1 = np.fft.rfft(a_x1).__abs__()
ft_a_x2 = np.fft.rfft(a_x2).__abs__()

ft_a_x1 /= ft_a_x2.max()
ft_a_x2 /= ft_a_x2.max()
rfreq = np.fft.rfftfreq(len(a_x1), 1e-9) * 1e-6

# %% fig 1
plt.figure()
plt.plot(rfreq, ft_a_x1, label="apodized unfiltered")
plt.plot(rfreq, ft_a_x2, label="apodized filtered")
plt.ylim(0.0, 1.05)
plt.xlabel("MHz")

# %% __________________________________________________________________________
full = np.load("data/phase_corrected/bckgnd/full.npy")
_1e4 = np.load("data/phase_corrected/bckgnd/1e4_pts.npy")
_1e3 = np.load("data/phase_corrected/bckgnd/1e3_pts.npy")

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
_1e4 = np.load("data/phase_corrected/su8/1e4_pts.npy")
_1e3 = np.load("data/phase_corrected/su8/1e3_pts.npy")

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
avg = np.load("data/phase_corrected/"
              "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
              mmap_mode='r')
avg = np.mean(avg, 0)
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
avg = np.load("data/phase_corrected/"
              "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
              mmap_mode='r')
avg = np.mean(avg, 0)
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
