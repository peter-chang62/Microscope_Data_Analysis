# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.signal as si
from tqdm import tqdm
from numpy import ma
import scipy.constants as sc
from scipy.interpolate import InterpolatedUnivariateSpline
import os

cr.style_sheet()

# %% global variables
# together
list_filter = np.array(
    [
        [0, 2],
        [31, 32.8],
        [249.5, 250.5],
        [280, 284],
        [337, 338],
        [436, 437],
        [499, 500],
    ]
)


# %% function defs
def filt(freq, ll, ul, x, type="bp"):
    if type == "bp":
        mask = np.where(np.logical_and(freq > ll, freq < ul), 0, 1)
    elif type == "notch":
        mask = np.where(np.logical_or(freq < ll, freq > ul), 0, 1)

    return ma.array(x, mask=mask)


def apply_filter(ft, lst_fltrs=list_filter):
    rfreq = np.fft.rfftfreq((len(ft) - 1) * 2, 1e-9) * 1e-6
    for f in lst_fltrs:
        ft = filt(rfreq, *f, ft, "notch")

    return ft


def calculate_snr(data, apod=None, avg_f=None):
    ppifg = data[0].size
    center = ppifg // 2

    freq_f = np.fft.rfftfreq(len(data[0]))  # 1 GHz frequency axis

    if avg_f is None:
        avg_f = np.mean(data, 0)
        avg_f -= np.mean(avg_f)
    ft_f = np.fft.rfft(avg_f)

    if not np.any([apod is None, apod == np.nan, ma.is_masked(apod)]):
        print("apodizing data")
        data = data[:, center - apod // 2 : center + apod // 2]
    else:
        print("NOT apodizing data")
    freq_a = np.fft.rfftfreq(len(data[0]))  # apodized frequency axis

    b, a = si.butter(4, 0.2, "low")
    amp_ft_f_filt = si.filtfilt(b, a, abs(ft_f))  # filter 1 GHz spectrum
    # interpolate 1 GHz spectrum onto the coarser frequency axis
    amp_ft_f_filt_gridded = InterpolatedUnivariateSpline(freq_f, amp_ft_f_filt)
    amp_ft_f_filt_interp = amp_ft_f_filt_gridded(freq_a)
    # look at signal range of interest
    ll_a = np.argmin(abs(freq_a - 0.10784578053383662))
    ul_a = np.argmin(abs(freq_a - 0.19547047721757888))
    denom_f = amp_ft_f_filt_interp[ll_a:ul_a]

    x = 0
    NOISE = np.zeros(len(data))
    for n, i in enumerate(tqdm(data)):
        i = i - np.mean(i)

        ft = np.fft.rfft(i)
        x = (x * n + apply_filter(ft)) / (n + 1)

        num = x.__abs__()[ll_a:ul_a]
        absorption = num / denom_f
        absorbance = -np.log(absorption)
        noise = np.std(absorbance)
        NOISE[n] = noise

    return NOISE


# %%
if os.name == "nt":
    path = (
        r"C:\Users\pchan\SynologyDrive\Research_Projects\Microscope/"
        r"Python_Workspace\data\phase_corrected/"
    )
else:
    path = (
        r"/Volumes/Peter SSD/Research_Projects/Microscope/Python_Workspace"
        r"/data/phase_corrected/"
    )

# %%
# =============================================================================
# data = np.load(  # taken on silicon
#     path + "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#     mmap_mode="r",
# )
# avg = np.load(path + "bckgnd/avg_bckgnd.npy")
#
# data = np.load(  # taken on su8
#     path + "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
#     mmap_mode="r",
# )
# avg = np.load(path + "su8/avg_su8.npy")
#
# ppifg = len(data[0])
# center = ppifg // 2
#
# resolution = np.arange(0, 500 + 10, 10)
# resolution[0] = 1
# APOD = (1 / resolution) * ppifg
# APOD = np.round(APOD).astype(int)
# APOD = np.where(APOD % 2 == 0, APOD, APOD + 1)
#
# APOD = ma.asarray(APOD)
# APOD[0] = ma.masked
#
# SIGMA = np.zeros((len(APOD), len(data)))
# for n, apod in enumerate(APOD):
#     SIGMA[n] = calculate_snr(data, apod, avg)
#     print(f"_____________________{len(APOD) - n - 1}_____________________")
#
# np.save(path + "su8/sigma/sigma.npy", SIGMA)
# =============================================================================

# %%
s_su8 = np.load(path + "su8/sigma/sigma.npy")
s_bckgnd = np.load(path + "bckgnd/sigma/sigma.npy")
window = np.load(path + "su8/sigma/NPTS.npy")
ppifg = 74180
center = ppifg // 2

n_ifg = np.arange(1, len(s_bckgnd[0]) + 1)
s_bckgnd_dB = 10 * np.log10(s_bckgnd)
s_su8_dB = 10 * np.log10(s_su8)

snr_bckgnd_dB = 10 * np.log10(1 / s_bckgnd)
snr_su8_dB = 10 * np.log10(1 / s_su8)

resolution = window[0] / window
resolution = np.round(resolution, 0)

# snr 2D plots
fig = plt.figure()
plt.suptitle("background absorbance noise (dB)")
plt.pcolormesh(n_ifg, resolution, snr_bckgnd_dB, cmap="jet")
plt.xscale("log")
plt.xlabel("# interferograms")
plt.ylabel("resolution (GHz)")
plt.colorbar()

fig = plt.figure()
plt.suptitle("su8 absorbance noise (dB)")
plt.pcolormesh(n_ifg, resolution, snr_su8_dB, cmap="jet")
plt.xscale("log")
plt.xlabel("# interferograms")
plt.ylabel("resolution (GHz)")
plt.colorbar()

# %% # create a gif
# target = "su8"
# save = False

# if target == "su8":
#     avg = np.load(path + "su8/avg_su8.npy")
#     data = np.load(
#         path + "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
#         mmap_mode="r",
#     )
#     snr = snr_su8_dB
# elif target == "bckgnd":
#     avg = np.load(path + "bckgnd/avg_bckgnd" ".npy")
#     data = np.load(
#         path + "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#         mmap_mode="r",
#     )
#     snr = snr_bckgnd_dB
# avg -= np.mean(avg)
# avg_1000 = np.mean(data[:1000], axis=0)

# ft_f = np.fft.rfft(avg)
# f_f = np.fft.rfftfreq(avg.size, d=1e-9) * ppifg
# f_ll, f_ul = 0.10784578053383662, 0.19547047721757888
# f_ll = f_ll * ppifg * 1e9 + f_f[-1] * 2
# f_ul = f_ul * ppifg * 1e9 + f_f[-1] * 2
# f_f += f_f[-1] * 2  # 3rd Nyquist window
# b, a = si.butter(4, 0.2, "low")
# amp_ft_f_filt = si.filtfilt(b, a, abs(ft_f))  # filter 1 GHz spectrum

# fig, ax = plt.subplots(1, 2, figsize=np.array([10.98, 4.8]))
# for n in range(resolution.size):
#     npts = window[n]
#     avg_a = avg_1000[center - npts // 2 : center + npts // 2]

#     ft_a = np.fft.rfft(avg_a)
#     f_a = np.fft.rfftfreq(avg_a.size, d=1e-9) * ppifg
#     f_a += f_a[-1] * 2  # 3rd Nyquist window
#     wl_a = sc.c / f_a
#     amp_ft_f_filt_gridded = InterpolatedUnivariateSpline(f_f, amp_ft_f_filt)
#     amp_ft_f_filt_interp = amp_ft_f_filt_gridded(f_a)

#     [i.clear() for i in ax]
#     ax[0].plot(wl_a * 1e6, amp_ft_f_filt_interp, label="background")
#     ax[0].plot(wl_a * 1e6, abs(ft_a), label="signal")
#     ax[0].axvline(sc.c * 1e6 / f_ll, color="r", linestyle="--")
#     ax[0].axvline(sc.c * 1e6 / f_ul, color="r", linestyle="--")
#     ax[1].semilogx(n_ifg, snr[0], "o")
#     ax[1].semilogx(n_ifg, snr[n], "o")

#     ax[0].legend(loc="best")
#     ax[0].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
#     ax[1].set_xlabel("interferogram #")
#     ax[0].set_ylabel("power spectrum (a.u.)")
#     ax[1].set_ylabel("SNR (dB)")
#     ax[1].axvline(1000, color="r", linestyle="--")
#     fig.suptitle(f"{int(resolution[n])} GHz resolution")

#     if save:
#         plt.savefig(f"fig/{n}.png")
#     else:
#         plt.pause(0.1)
