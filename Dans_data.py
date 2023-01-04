"""" optimiztion in the time domain doesn't work! It never gives an answer
different from the initial guess, even when it's obviously at an unstable
maximum. """

# %%
import sys
sys.path.append("include/")
import numpy as np
import phase_correction as pc
import matplotlib.pyplot as plt
import td_phase_correct as td
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()
plt.ion()

# %% _________________________________ load the data __________________________
path = r"data/Dans_interferograms/"
data = np.genfromtxt(path + "Data.txt")
T = np.genfromtxt(path + "t_axis.txt")
N = np.arange(-len(data[0]) // 2, len(data[0]) // 2)
ppifg = len(N)
center = ppifg // 2

# %% _________________________________ phase correction _______________________
opt = td.Optimize(data)
opt.phase_correct(1500)

# %% _______________________________________ plotting _________________________
avg = np.mean(opt.CORR, axis=0)
ft = abs(pc.fft(avg))

plt.figure()
plt.plot(ft[center:])

plt.figure()
plt.plot(avg)
