"""
This uses run.npy and run_sample.npy that's generated by
bio_static_reprocess.py
"""

# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import clipboard


try:
    import mkl_fft

    rfft_numpy = mkl_fft.rfft_numpy
    irfft_numpy = mkl_fft.irfft_numpy
except ImportError:
    rfft_numpy = np.fft.rfft
    irfft_numpy = np.fft.irfft


def rfft(x, axis=-1):
    return rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(irfft_numpy(x, axis=axis), axes=axis)


ppifg = 77760
center = ppifg // 2

# %% ----- apodization
# resolution = 5
# apod = ppifg // resolution
# apod = apod if apod % 2 == 0 else apod + 1

# run = np.load("run.npy")
# run[np.isnan(run)] = 0

# index = np.arange(0, len(run), 250)
# index[-1] = run.shape[0]
# run_a = np.zeros((run.shape[0], len(np.fft.rfftfreq(apod))))
# print("calculating apodized data!")
# for n in tqdm(range(len(index) - 1)):
#     t = irfft(run[index[n] : index[n + 1]])
#     t = t[:, center - apod // 2 : center + apod // 2]
#     run_a[index[n] : index[n + 1]] = rfft(t)

# # np.save("run_a.npy", run_a)

# # %% -----
# # run_a = np.load("run_a.npy")

# f = np.fft.rfftfreq(apod)
# f_ll, f_ul = 0.0660088483243921, 0.21660454260433362
# (ind,) = np.logical_and(f_ll < f, f < f_ul).nonzero()

# # %% -----
# absorb_a = run_a[:, ind]
# absorb_a = -np.log(absorb_a / absorb_a[-1])
# snr_a = np.std(absorb_a, axis=1)

# %% ----- for loop the larger apodization windows!
# run = np.load("run.npy")
# run[np.isnan(run)] = 0

resolution = np.logspace(np.log10(1), np.log10(100), num=100)
apod = ppifg // resolution
apod[apod % 2 == 1] += 1
apod = apod.astype(int)

# SNR = np.zeros((apod.size, run.shape[0]))
# index = np.arange(0, len(run), 250)
# index[-1] = run.shape[0]
# for m, a in enumerate(tqdm(apod)):
#     if a == ppifg:
#         run_a = run
#     else:
#         run_a = np.zeros((run.shape[0], len(np.fft.rfftfreq(a))))
#         print("calculating apodized data!")
#         for n in tqdm(range(len(index) - 1)):
#             t = irfft(run[index[n] : index[n + 1]])
#             t = t[:, center - a // 2 : center + a // 2]
#             run_a[index[n] : index[n + 1]] = abs(rfft(t))

#     f = np.fft.rfftfreq(a)
#     f_ll, f_ul = 0.0660088483243921, 0.21660454260433362
#     (ind,) = np.logical_and(f_ll < f, f < f_ul).nonzero()

#     absorb_a = run_a[:, ind]
#     absorb_a = -np.log(absorb_a / absorb_a[-1])
#     snr_a = np.std(absorb_a, axis=1)

#     SNR[m] = snr_a

# np.save("SNR_2D.npy", SNR)

# %% ----- gif showing how the snr increases with apodization
path = r"/Volumes/Peter SSD/Physical_Cloud/Microscope_Data_Analysis/CLEO_2023/"
run = np.load(path + "run.npy", mmap_mode="r")
run_bio = np.load(path + "run_sample.npy", mmap_mode="r")
t = run[500].copy()  # after 500 averages
t_b = run_bio[500].copy()
t[np.isnan(t)] = 0
t_b[np.isnan(t_b)] = 0

t = irfft(t)
t_b = irfft(t_b)
save = False
figsize = np.array([4.64, 3.63])
fig, ax = plt.subplots(1, 1, figsize=figsize)
for n, a in enumerate(tqdm(apod)):
    ft = abs(rfft(t[center - a // 2 : center + a // 2]))
    ft_b = abs(rfft(t_b[center - a // 2 : center + a // 2]))
    absorb = -np.log(ft_b / ft)

    f = np.fft.rfftfreq(a, d=1e-3) * ppifg
    f += f[-1] * 2
    wl = 299792458 / f
    (ind,) = np.logical_and(3.34 < wl, wl < 3.577).nonzero()

    ax.clear()
    ax.plot(
        wl[ind],
        absorb[ind],
        # label=f"{np.round(resolution[n], 2)} GHz",
    )
    ax.set_ylim(0.17062564355116486 - 0.07, 0.2977848306378814 + 0.07)
    ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
    ax.set_ylabel("absorbance")
    ax.set_title(f"{np.round(resolution[n], 2)} GHz", fontsize=18)
    ax.axis(False)
    # ax.legend(loc="best", fontsize=16)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()

    if save:
        plt.savefig(f"../fig/{n}.png", dpi=300, transparent=True)
    else:
        plt.pause(0.05)
