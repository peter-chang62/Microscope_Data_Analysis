import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import td_phase_correct as td
import scipy.signal as ss
import digital_phase_correction as dpc

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
data = data[:2000]

opt = td.Optimize(data)
opt.phase_correct(20)

opt_apod = td.Optimize(data[:, center - 1000:center + 1000])
opt_apod.phase_correct(20)

avg = np.mean(opt.CORR, 0)
amp_avg = dpc.fft(avg).__abs__()

avg_apod = np.mean(opt_apod.CORR, 0)
amp_avg_apod = dpc.fft(avg_apod).__abs__()

freq = np.fft.fftshift(np.fft.fftfreq(len(amp_avg)))
freq_apod = np.fft.fftshift(np.fft.fftfreq(len(amp_avg_apod)))

plt.figure()
plt.plot(freq, amp_avg)
plt.plot(freq_apod, amp_avg_apod)
