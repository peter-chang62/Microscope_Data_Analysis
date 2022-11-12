import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt
from scipy.integrate import simps
from images import *
import os
import scipy.constants as sc
import phase_correction as pc

if os.name == 'posix':
    path = r"/home/peterchang/SynologyDrive/Research_Projects/Microscope/CLEO_2023/data_to_plot/"
else:
    path = r"C:\Users\pchan\SynologyDrive\Research_Projects\Microscope\CLEO_2023\data_to_plot/"

cr.style_sheet()

x_group3, y_group3, i_group3, s_group3 = group_3(False)
x_num4, y_num4, i_num4, s_num4 = num_4(False)
x_bar, y_bar, i_bar, s_bar = smallest_bar(False)

ind_inbw_patt = [57, 35]
ind_inbw_bar = [30, 27]
ind_on_bar = [24, 19]

snr_bckgnd = np.load(path + "snr_bckgnd.npy")
snr_su8 = np.load(path + "snr_su8.npy")
avg_bckgnd = np.load(path + "avg_bckgnd.npy")
avg_su8 = np.load(path + "avg_su8.npy")

frep = 1e9
ppifg = 74180
center = ppifg // 2
Nyq_freq = frep * center
nu = np.linspace(0, Nyq_freq, center) + Nyq_freq * 2
wl = sc.c / nu * 1e6

bckgnd = abs(pc.fft(avg_bckgnd)[center:])
su8 = abs(pc.fft(avg_su8)[center:])

# ______________________________________________________ plotting ______________________________________________________
cmap = 'cividis'

# group 3
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(x_group3, y_group3, i_group3, cmap=cmap)
ax.set_xlabel("$\mathrm{\mu m}$")
ax.set_ylabel("$\mathrm{\mu m}$")
cr.square()

# smallest patterns in group 3
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(x_bar, y_bar, i_bar, cmap=cmap)
cr.square()

# number 4
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(x_num4, y_num4, i_num4, cmap=cmap)
cr.square()

fig, ax = plt.subplots(1, 2)
ax[0].plot(wl, bckgnd / 26.5e3)
ax[0].set_ylim(0, 1)
ax[1].loglog(snr_bckgnd[:, 0], snr_bckgnd[:, 1], 'o')
ax[0].set_xlabel("wavelength ($\mathrm{\mu m}$)")
ax[1].set_xlabel("time (s)")
