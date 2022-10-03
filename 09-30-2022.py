import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import scipy.integrate as spi

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
path = r'/home/peterchang/SynologyDrive/Research_Projects/Microscope/LineScans/09-30-2022/'
spectra = np.load(path + 'line_scan_amps_in_loop_spectra.npy')
pos = np.load(path + "line_scan_amps_in_loop_stage_pos.npy")
pos -= pos.min()

center = len(spectra[0])
Nyq_freq = center * 1e9
Nyq_Window = 3
translation = (Nyq_Window - 1) * Nyq_freq
nu = np.linspace(0, Nyq_freq, center) + translation
wl = sc.c * 1e6 / nu

integral = spi.simps(spectra, x=nu * 1e-12, axis=1)

save = True
fig, ax = plt.subplots(1, 2)
for n, i in enumerate(spectra):
    [i.clear() for i in ax]
    ax[0].plot(wl, i)
    ax[0].set_ylim(0, 1)
    ax[1].plot(pos[:n], integral[:n])
    ax[1].set_ylim(0, integral.max())
    ax[1].set_xlim(pos.min(), pos.max())
    ax[0].set_xlabel("wavelength ($\mathrm{\mu m}$)")
    ax[0].set_xlabel("stage position ($\mathrm{\mu m}$)")
    if save:
        plt.savefig(f'fig/{n}.png')
    else:
        plt.pause(.01)
    print(len(spectra) - n - 1)
