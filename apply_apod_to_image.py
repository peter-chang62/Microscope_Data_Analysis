# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
from tqdm import tqdm
import scipy.constants as sc
import os

cr.style_sheet()

# %% global variables
if os.name == "nt":
    # this would be something else on the GaGe computer...
    path_folder = (
        r"C:\\Users\\pchan\\SynologyDrive\\Research_Projects\\Microscope\\Images/"
    )

else:
    # linux partition
    # path_folder = r"/Users/peterchang/SynologyDrive/Research_Projects" \
    #               r"/Microscope/Images/"
    path_folder = r"/Volumes/Peter SSD/Research_Projects/Microscope/Images/"

list_filter = np.array(
    [
        [0, 2],
        [249, 251],
        [279, 283],
        [333, 336],
        [438, 440],
        [499, 500],
    ]
)


# %% function defs
def filt(freq, ll, ul, x, type="bp"):
    if type == "bp":
        return np.where(np.logical_and(freq > ll, freq < ul), x, 0.0)

    elif type == "notch":
        return np.where(np.logical_or(freq < ll, freq > ul), x, 0.0)


def apply_filt(ft, lst_fltrs=list_filter):
    rfreq = np.fft.rfftfreq((len(ft) - 1) * 2, 1e-9) * 1e-6
    for f in lst_fltrs:
        ft = filt(rfreq, *f, ft, "notch")

    return ft


def num_4():
    path = path_folder + "11-08-2022/"
    s = np.load(
        path + "stage1_5932_6066_stage2_8478_8575p5_step_2p5_ppifg_74180.npy",
        mmap_mode="r",
    )
    s = np.transpose(s, axes=[1, 0, 2])
    x = np.arange(5932, 6066, 2.5)
    y = np.arange(8478, 8575.5 + 2.5, 2.5)

    return x, y, s


def smallest_bar():
    path = path_folder + "11-08-2022/"
    s = np.load(
        path + "stage1_6274_6460_stage2_8593_8883_step_2p5_ppifg_74180.npy",
        mmap_mode="r",
    )
    s = np.transpose(s, axes=[1, 0, 2])
    x = np.arange(6274, 6460, 2.5)
    y = np.arange(8593, 8883 + 2.5, 2.5)

    return x, y, s


# %%
x, y, s = num_4()
shape = s.shape
s = s.reshape((s.shape[0] * s.shape[1], s.shape[2]))
for n, i in enumerate(tqdm(s)):
    s[n] = apply_filt(i)
s = s.reshape(shape)
t = np.fft.ifftshift(np.fft.irfft(s, axis=-1), axes=-1)

resolution = 100
ppifg = 74180
center = ppifg // 2
window = int(np.round(ppifg / resolution))
t_a = t[:, :, center - window // 2 : center + window // 2]
s_a = np.fft.rfft(np.fft.fftshift(t_a, axes=-1), axis=-1)
s_a = abs(s_a)

# %%
absorption = s / s[0, 0]
absorbance = -np.log(absorption)

absorption_a = s_a / s_a[0, 0]
absorbance_a = -np.log(absorption_a)

# %%
nu = np.fft.rfftfreq(ppifg - 1, 1e-9) * ppifg
nu += nu[-1] * 2
wl = sc.c * 1e6 / nu
wl_ll, wl_ul = 3.25, 3.6
ind_ll, ind_ul = np.argmin(abs(wl - wl_ul)), np.argmin(abs(wl - wl_ll))

img = simpson(absorbance[:, :, ind_ll:ind_ul])
img -= img.min()
img *= -1

# %%
nu_a = np.fft.rfftfreq(window, 1e-9) * ppifg
nu_a += nu_a[-1] * 2
wl_a = sc.c * 1e6 / nu_a
ind_ll_a, ind_ul_a = np.argmin(abs(wl_a - wl_ul)), np.argmin(abs(wl_a - wl_ll))

img_a = simpson(absorbance_a[:, :, ind_ll_a:ind_ul_a])
img_a -= img_a.min()
img_a *= -1

# %% plotting
plt.figure()
plt.plot(np.fft.rfftfreq(ppifg - 1), s[0, 0])
plt.plot(np.fft.rfftfreq(window), s_a[0, 0])
plt.plot(np.fft.rfftfreq(ppifg - 1), s[13, 27])
plt.plot(np.fft.rfftfreq(window), s_a[13, 27])

plt.figure()
plt.pcolormesh(img)
plt.title("unapodized image")

plt.figure()
plt.pcolormesh(img_a)
plt.title("apodized image")

plt.figure()
plt.plot(wl, absorbance[13, 27])
plt.plot(wl_a, absorbance_a[13, 27])
[plt.axvline(i, color="r") for i in [wl_ll, wl_ul]]
plt.ylim(0.6, 1.9)
plt.xlim(3.2, 3.65)
