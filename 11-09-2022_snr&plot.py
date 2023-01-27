import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.signal as si
from tqdm import tqdm
from numpy import ma

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


def calculate_snr(data, apod=None):
    ppifg = len(data[0])
    center = ppifg // 2

    if not np.any([apod is None, apod == np.nan, ma.is_masked(apod)]):
        # assert isinstance(apod, (int, np.int64)), "apod must be an integer"
        print("apodizing data")
        data = data[:, center - apod // 2:center + apod // 2]
    else:
        print("NOT apodizing data")
    freq = np.fft.rfftfreq(len(data[0]))

    avg = np.mean(data, 0)
    avg -= np.mean(avg)
    ft_avg = np.fft.rfft(avg)

    ll = np.argmin(abs(freq - .10784578053383662))
    ul = np.argmin(abs(freq - .19547047721757888))

    b, a = si.butter(4, .2, "low")
    ft_avg_filt = si.filtfilt(b, a, ft_avg.__abs__())
    denom = ft_avg_filt[ll:ul]

    x = 0
    NOISE = np.zeros(len(data))
    for n, i in enumerate(tqdm(data)):
        i = i - np.mean(i)

        ft = np.fft.rfft(i)
        x = (x * n + apply_filter(ft)) / (n + 1)

        num = x.__abs__()[ll:ul]
        absorption = num / denom
        absorbance = -np.log(absorption)
        noise = np.std(absorbance)
        NOISE[n] = noise

    return NOISE


# %% __________________________________________________________________________
# data = np.load(  # taken on silicon background
#     "data/phase_corrected/stage1_5116_stage2_8500_53856x74180_phase_corrected.npy",
#     mmap_mode='r')
data = np.load(  # taken on su8
    "data/phase_corrected/stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
    mmap_mode='r')
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
    SIGMA[n] = calculate_snr(data, apod)
    print(f'_____________________{len(APOD) - n - 1}_____________________')

np.save("data/phase_corrected/su8/sigma/sigma.npy", SIGMA)

# %% __________________________________________________________________________
# sigma_su8 = np.load("data/phase_corrected/su8/sigma/sigma.npy")
# sigma_bckgnd = np.load("data/phase_corrected/bckgnd/sigma/sigma.npy")
# window = np.load("data/phase_corrected/su8/sigma/NPTS.npy")
# ppifg = 74180
# center = ppifg // 2

# n_ifg = np.arange(1, len(sigma_bckgnd[0]) + 1)
# s_bckgnd_dB = 10 * np.log10(sigma_bckgnd)
# s_su8_dB = 10 * np.log10(sigma_su8)

# resolution = window[0] / window
# resolution = np.round(resolution, 0)

# fig = plt.figure()
# plt.suptitle("background absorbance noise (dB)")
# plt.pcolormesh(n_ifg, resolution, s_bckgnd_dB, cmap='jet')
# plt.xscale('log')
# plt.xlabel("# interferograms")
# plt.ylabel("resolution (GHz)")
# plt.colorbar()

# fig = plt.figure()
# plt.suptitle("su8 absorbance noise (dB)")
# plt.pcolormesh(n_ifg, resolution, s_su8_dB, cmap='jet')
# plt.xscale('log')
# plt.xlabel("# interferograms")
# plt.ylabel("resolution (GHz)")
# plt.colorbar()

# # create a gif showing how the absorbance noise changes with
# # apodization window
# fig, ax = plt.subplots(1, 2, figsize=np.array([10.64, 4.8]))
# avg = np.load("data/phase_corrected/bckgnd/avg_bckgnd.npy")
# freq_full = np.fft.rfftfreq(len(avg))
# s_full = apply_filter(np.fft.rfft(avg).__abs__())
# ind_full = np.logical_and(freq_full > 0.10784578053383662,
#                           freq_full < 0.19547047721757888).nonzero()[0]
# save = True
# for h, w in enumerate(window):
#     x = avg[center - w // 2:center + w // 2]
#     s = apply_filter(np.fft.rfft(x).__abs__())
#     freq = np.fft.rfftfreq(len(x))
#     ind = np.logical_and(freq > 0.10784578053383662,
#                          freq < 0.19547047721757888).nonzero()[0]
#
#     [i.clear() for i in ax]
#     ax[0].plot(freq_full[ind_full], s_full[ind_full])
#     ax[0].plot(freq[ind], s[ind])
#     ax[1].loglog(sigma_bckgnd[0], 'o', label="1 GHz")
#     ax[1].loglog(sigma_bckgnd[h], 'o',
#                  label=f'{np.round(74180 / window[h], 1)} GHz')
#     ax[1].set_ylim(
#         0.0005,  # bckgnd
#         # 0.0017, # su8
#         0.8)
#     ax[1].legend(loc='best')
#     if save:
#         plt.savefig(f'fig/{h}.png')
#     else:
#         plt.pause(.05)
#
#     print(len(window) - h - 1)
