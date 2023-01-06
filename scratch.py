import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import scipy.signal as ss
import digital_phase_correction as dpc
import scipy.optimize as so
import scipy.interpolate as si

cr.style_sheet()

path = "/Users/peterchang/SynologyDrive/Research_Projects/" \
       "Microscope/FreeRunningSpectra/11-09-2022/"

data = np.load(path + "stage1_5116_stage2_8500_53856x74180.npy",
               mmap_mode='r')

ppifg = len(data[0])
center = ppifg // 2
N_IFG = len(data)

data.resize((ppifg * N_IFG,))
data = data[center:-center]
data.resize((N_IFG - 1, ppifg))
data = data[::50]

# %% __________________________________________________________________________
# opt = td.Optimize(data[:, center - 10:center + 10])
# corr = data.copy()
# opt.phase_correct(corr)

# %% __________________________________________________________________________
subst = data[:, :1000]
opt = td.Optimize(subst)
corr = data.copy()
opt.phase_correct(corr)

# %%
avg = np.mean(corr, 0)
bckgnd = np.hstack([avg[:center - 5], avg[center + 5:]])
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(bckgnd))),
         dpc.fft(bckgnd).__abs__())
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(avg))), dpc.fft(avg).__abs__())
plt.ylim(0, 1.2e4)
plt.xlim(0, .4)

plt.show()
