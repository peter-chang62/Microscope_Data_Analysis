# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from include import digital_phase_correction as dpc
from tqdm import tqdm

# %% load the data
path = r"Data/03-8-2023/73728_is_a_fast_fft_length_54180x73728.npy"
ppifg = 73728
center = ppifg // 2

data = np.load(path, mmap_mode="r")
N_ifg = data.shape[0]
data.resize(data.size)
data = data[center:-center]
data.resize((N_ifg - 1, ppifg))

# %%
# np.save("Data/_overwrite/_overwrite.npy", data)
_overwrite = np.load("Data/_overwrite/_overwrite.npy", mmap_mode="r+")

# -------------- TRY 1: just zero the phase -----------------------------------
step = len(data) // 8
n = np.arange(0, len(data), step)
n[-1] = len(data)
start = n[:-1]
end = n[1:]
console = 7
for n, x in enumerate(tqdm(data[start[console] : end[console]])):
    n += start[console]

    x = np.fft.ifftshift(x)

    # filter out f0 beats
    # needed to eliminate ringing when apodizing to fit the spectral phase
    ft = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), d=1e-9)
    filters = np.array(
        [
            [-1, 1],
            [277, 280],
            [330, 334],
            [442, 444],
            [499, 501],
        ]
    )
    for i in filters * 1e6:
        ind = np.logical_and(i[0] < freq, freq < i[1])
        ft[ind] = 0

    # -------------------------------------------------------------------------

    # get the spectral phase of the apodized interferogram
    x_f = np.fft.fftshift(np.fft.irfft(ft))
    f, p, a = dpc.get_phase(x_f, 400, plot=False)

    # get the region to fit the phase to a quadratic
    ll, ul = 60, 107  # determined this range from plotting
    ind_center = int((ul - ll) / 2 + ll)
    f_center = f[ind_center]
    f -= f_center

    # fit to a quadratic
    polyfit = np.polyfit(f[ll:ul], p[ll:ul], 2)
    polyfit = polyfit[1:]  # don't use the quadratic component
    poly1d = np.poly1d(polyfit)

    # subtract out the quadratic phase
    f_full = np.fft.rfftfreq(len(x)) - f_center
    ft *= np.exp(-1j * np.poly1d(poly1d)(f_full))

    # -------------------------------------------------------------------------

    # save it
    x_f_p = np.fft.fftshift(np.fft.irfft(ft))
    _overwrite[n] = x_f_p

# %% -------- TRY 2 -------
# from include import td_phase_correct as td

# # %%
# data = np.load("Data/_overwrite/_overwrite.npy", "r")
# data_to_shift = np.load("Data/_overwrite/_overwrite_2.npy", "r+")
# ppifg = len(data[0])
# center = ppifg // 2

# step = len(data) // 8
# n = np.arange(0, len(data), step)
# n[-1] = len(data)
# start = n[:-1]
# end = n[1:]

# # %%
# console = 7
# opt = td.Optimize(data[:, center - 25 : center + 25])
# opt.phase_correct(
#     data_to_shift=data_to_shift,
#     overwrite_data_to_shift=True,
#     start_index=start[console],
#     end_index=end[console],
#     method="Powell",
# )
