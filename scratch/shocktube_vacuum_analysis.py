import sys

sys.path.append("../include/")
import numpy as np
from tqdm import tqdm
import os
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import matplotlib.pyplot as plt
import digital_phase_correction as dpc
from numpy import ma
from scipy.integrate import simps

path = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022" \
       r"\Vacuum_Background_end_of_experiment/"
path_h2co = path + "3.5um_filter_114204x17506.bin"
path_co = path + "4.5um_filter_114204x17506.bin"

ppifg = 17506
center = ppifg // 2

h2co = np.fromfile(path_h2co, '<h')
h2co = h2co[np.argmax(h2co[:ppifg]) + center:]
h2co = h2co[:(len(h2co) // ppifg) * ppifg]
h2co.resize((len(h2co) // ppifg, ppifg))

ft_h2co = np.zeros((len(h2co), len(np.fft.rfftfreq(len(h2co[0])))),
                   dtype=complex)
step = 250
for h in tqdm(range(0, len(h2co), step)):
    ft_h2co[h: h + step] = np.fft.rfft(h2co[h: h + step], axis=1)

list_filter = np.array([
    [4350, 4400],
    [4440, 4853],
    [5200, 5400],
])
ft_h2co_filt = ft_h2co.copy()
for filt in list_filter:
    ft_h2co_filt[:, filt[0]:filt[1]] = 0

h2co_filt = np.zeros(h2co.shape)
step = 250
for h in tqdm(range(0, len(h2co), step)):
    h2co_filt[h: h + step] = np.fft.irfft(ft_h2co_filt[h: h + step], axis=1)

for n, i in enumerate(tqdm(h2co_filt)):
    h2co_filt[n] = np.roll(i, center - np.argmax(i))

opt = td.Optimize(h2co_filt[:, center - 20:center + 20])
opt.phase_correct(h2co_filt)

avg = np.mean(h2co_filt, 0)
np.save("../check_this_avg_out.npy", avg)
old_avg = np.load(path + "PHASE_CORRECTED/h2co_vacuum_bckgnd_avg.npy")

new_ft = np.fft.rfft(avg)
old_ft = np.fft.rfft(old_avg)

old_ft_filt = ma.array(data=old_ft)
new_ft_filt = ma.array(data=new_ft)
for filt in list_filter:
    old_ft_filt[filt[0]:filt[1]] = ma.masked
    new_ft_filt[filt[0]:filt[1]] = ma.masked

new_ft_filt = new_ft_filt[50:8600]
area_n = simps(abs(new_ft_filt))
area_o = simps(abs(old_ft_filt))
old_ft_filt = old_ft_filt[50:8600] * area_n / area_o
plt.figure()
plt.plot(abs(new_ft_filt), label="filtered + time domain")
plt.plot(abs(old_ft_filt), label="NOT filtered + frequency domain")
plt.legend(loc='best')
