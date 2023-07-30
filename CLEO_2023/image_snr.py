# %% -----
import matplotlib.pyplot as plt
import clipboard
import numpy as np
from scipy.integrate import simpson
import tables


# %% --------------------------------------------------------------------------
file = tables.open_file("bio_sample_100GHz_coarse.h5", "r")
data = file.root.data
absorbance_c = file.root.absorbance
img_c = absorbance_c[:, :, 97]

xlim = 288, 322
ylim = 263, 295
bckgnd_c = img_c[ylim[0] : ylim[1], xlim[0] : xlim[1]]

std_c = np.std(bckgnd_c)
snr_c = 1 / std_c

# %% --------------------------------------------------------------------------
file = tables.open_file("su8_sample_100GHz_fine.h5", "r")
data = file.root.data
absorbance_f_usaf = file.root.absorbance

img_f_usaf = absorbance_f_usaf[:, :, 100]

ylim = 95, 108
xlim = 178, 214

bckgnd_f_usaf = img_f_usaf[ylim[0] : ylim[1], xlim[0] : xlim[1]]

std_f_usaf = np.std(bckgnd_f_usaf)
snr_f_usaf = 1 / std_f_usaf
