import sys

sys.path.append("../include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import digital_phase_correction as dpc
import phase_correction as pc

clipboard_and_style_sheet.style_sheet()

# ________________________________________________ path and load data __________________________________________________
path = r"../data/Dans_interferograms/"
data = np.genfromtxt(path + "Data.txt")
T = np.genfromtxt(path + "t_axis.txt")
N = np.arange(-len(data[0]) // 2, len(data[0]) // 2)
ppifg = len(N)
center = ppifg // 2

# ________________________________________________ apodization window __________________________________________________
apod = 5000

# plt.plot(N, data[0])
# plt.axvline(-apod // 2)
# plt.axvline(apod // 2)

# ________________________________________________ fitting phase, find frequency fit window ____________________________
f, p, a, ax, ax2 = dpc.get_phase(data[np.random.randint(0, len(data))], apod)
dpc.get_phase(data[np.random.randint(0, len(data))], apod, ax=ax, ax2=ax2)
dpc.get_phase(data[np.random.randint(0, len(data))], apod, ax=ax, ax2=ax2)

ll, ul = .200526, .202436
# ll, ul = .20623, .20792

plt.axvline(ll, color='r')
plt.axvline(ul, color='r')

# ________________________________________________ phase correction ____________________________________________________
pdiff = dpc.get_pdiff(data, ll, ul, apod)
dpc.apply_t0_and_phi0_shift(pdiff, data)

# ________________________________________________ plot results  _______________________________________________________
plt.figure()
avg = np.mean(data, axis=0)
ft = dpc.fft(avg)
plt.plot(abs(ft))

plt.figure()
[plt.plot(i[center - 150 // 2:center + 300 // 2]) for i in data]

# ________________ same thing, but just do t0 correction via cross-correlation instead _________________________________
data_ = np.genfromtxt(path + "Data.txt")
data_ = (data.T - np.mean(data_, axis=1)).T
data_, shift = pc.t0_correct_via_cross_corr(data_, 5000, False)

plt.figure()
avg_2 = np.mean(data_, axis=0)
ft = dpc.fft(avg_2)
plt.plot(abs(ft))

plt.figure()
[plt.plot(i[center - 150 // 2:center + 300 // 2]) for i in data_]

# for sublime-text
plt.show()
