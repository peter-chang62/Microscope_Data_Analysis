import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.signal as si
from tqdm import tqdm
from numpy import ma
from scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing.pool import Pool

cr.style_sheet()

# together
list_filter = np.array([[0, 2],
                        [31, 32.8],
                        [249.5, 250.5],
                        [280, 284],
                        [337, 338],
                        [436, 437],
                        [499, 500]])


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
    ppifg = len(data[0])
    center = ppifg // 2

    freq_f = np.fft.rfftfreq(len(data[0]))  # 1 GHz frequency axis

    if avg_f is None:
        avg_f = np.mean(data, 0)
        avg_f -= np.mean(avg_f)
    ft_f = np.fft.rfft(avg_f)

    if not np.any([apod is None, apod == np.nan, ma.is_masked(apod)]):
        print("apodizing data")
        data = data[:, center - apod // 2:center + apod // 2]
    else:
        print("NOT apodizing data")
    freq_a = np.fft.rfftfreq(len(data[0]))  # apodized frequency axis

    b, a = si.butter(4, .2, "low")
    amp_ft_f_filt = si.filtfilt(b, a, abs(ft_f))  # filter 1 GHz spectrum
    # interpolate 1 GHz spectrum onto the coarser frequency axis
    amp_ft_f_filt_gridded = InterpolatedUnivariateSpline(freq_f, amp_ft_f_filt)
    amp_ft_f_filt_interp = amp_ft_f_filt_gridded(freq_a)
    # look at signal range of interest
    ll_a = np.argmin(abs(freq_a - .10784578053383662))
    ul_a = np.argmin(abs(freq_a - .19547047721757888))
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


# %% __________________________________________________________________________
path = r"/Volumes/Extreme SSD/Research_Projects/Microscope/Python_Workspace" \
       r"/data/phase_corrected/"

# %% __________________________________________________________________________
# data = np.load(  # taken on silicon
#     path + "stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#     mmap_mode='r')
# avg = np.load("/Volumes/Extreme SSD/Research_Projects/Microscope"
#               "/Python_Workspace/data/phase_corrected/bckgnd/avg_bckgnd.npy")
data = np.load(  # taken on su8
    path + "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
    mmap_mode='r')
avg = np.load("/Volumes/Extreme SSD/Research_Projects/Microscope"
              "/Python_Workspace/data/phase_corrected/su8/avg_su8.npy")
ppifg = len(data[0])
center = ppifg // 2

resolution = np.arange(0, 500 + 10, 10)
resolution[0] = 1
APOD = (1 / resolution) * ppifg
APOD = np.round(APOD).astype(int)
APOD = np.where(APOD % 2 == 0, APOD, APOD + 1)

APOD = ma.asarray(APOD)
APOD[0] = ma.masked

SIGMA = np.zeros((len(APOD), len(data)))
for n, apod in enumerate(APOD):
    SIGMA[n] = calculate_snr(data, apod, avg)
    print(f'_____________________{len(APOD) - n - 1}_____________________')

np.save("sigma_su8.npy", SIGMA)

# %% __________________________________________________________________________
# s_su8 = np.load("sigma_su8.npy")
# s_bckgnd = np.load("sigma_bckgnd.npy")
# window = np.load(path + "su8/sigma/NPTS.npy")
# ppifg = 74180
# center = ppifg // 2
#
# n_ifg = np.arange(1, len(s_bckgnd[0]) + 1)
# s_bckgnd_dB = 10 * np.log10(s_bckgnd)
# s_su8_dB = 10 * np.log10(s_su8)
#
# snr_bckgnd_dB = 10 * np.log10(1 / s_bckgnd)
# snr_su8_dB = 10 * np.log10(1 / s_su8)
#
# resolution = window[0] / window
# resolution = np.round(resolution, 0)
#
# # snr 2D plots
# fig = plt.figure()
# plt.suptitle("background absorbance noise (dB)")
# plt.pcolormesh(n_ifg, resolution, snr_bckgnd_dB, cmap='jet')
# plt.xscale('log')
# plt.xlabel("# interferograms")
# plt.ylabel("resolution (GHz)")
# plt.colorbar()
#
# fig = plt.figure()
# plt.suptitle("su8 absorbance noise (dB)")
# plt.pcolormesh(n_ifg, resolution, snr_su8_dB, cmap='jet')
# plt.xscale('log')
# plt.xlabel("# interferograms")
# plt.ylabel("resolution (GHz)")
# plt.colorbar()
