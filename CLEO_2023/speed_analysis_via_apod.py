# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import mkl_fft
from tqdm import tqdm
import os


def rfft(x, axis=-1):
    return mkl_fft.rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(mkl_fft.irfft_numpy(x, axis=axis), axes=axis)


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
run = np.load("run.npy")
run[np.isnan(run)] = 0

resolution = np.logspace(np.log10(1), np.log10(100), num=100)
apod = ppifg // resolution
apod[apod % 2 == 1] += 1
apod = apod.astype(int)

SNR = np.zeros((apod.size, run.shape[0]))
index = np.arange(0, len(run), 250)
index[-1] = run.shape[0]
for m, a in enumerate(tqdm(apod)):
    if a == ppifg:
        run_a = run
    else:
        run_a = np.zeros((run.shape[0], len(np.fft.rfftfreq(a))))
        print("calculating apodized data!")
        for n in tqdm(range(len(index) - 1)):
            t = irfft(run[index[n] : index[n + 1]])
            t = t[:, center - a // 2 : center + a // 2]
            run_a[index[n] : index[n + 1]] = abs(rfft(t))

    f = np.fft.rfftfreq(a)
    f_ll, f_ul = 0.0660088483243921, 0.21660454260433362
    (ind,) = np.logical_and(f_ll < f, f < f_ul).nonzero()

    absorb_a = run_a[:, ind]
    absorb_a = -np.log(absorb_a / absorb_a[-1])
    snr_a = np.std(absorb_a, axis=1)

    SNR[m] = snr_a

np.save("SNR_2D.npy", SNR)

# %% ----- look at the results!
# names = [i.name for i in os.scandir(".") if "snr" in i.name]
# key = lambda s: int(s.split("_")[1].split("GHz")[0])
# names.sort(key=key)
# snr = np.vstack([np.load(i) for i in names])
