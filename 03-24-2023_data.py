"""
Something's still wrong ...
"""
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from include import digital_phase_correction as dpc
from tqdm import tqdm


def apply_filter(freq, ft, filt, d2=False):
    for f in filt:
        ll, ul = f
        (ind,) = np.logical_and(ll < freq, freq < ul).nonzero()
        if d2:
            ft[:, ind] = 0
        else:
            ft[ind] = 0


path = r"H:\\Research_Projects\\Microscope\\FreeRunningSpectra\\03-23-2023/"
data = np.load(path + "think_on_bio_sample_51408x77760.npy", mmap_mode="r")
N_ifg, ppifg = data.shape
center = ppifg // 2
data.resize(data.size)
data = data[center:-center]
data.resize((N_ifg - 1, ppifg))

filt = np.array(
    [
        [-1, 15],
        [249.8, 250.2],
        [275, 290],
        [330.3, 336.3],
        [436.8, 437.2],
        [499, 501],
    ]
)

# _oversrite = np.save("_overwrite.npy", np.zeros(data.shape))
_overwrite = np.load("_overwrite.npy", "r+")

apod = 400

for n, x in enumerate(tqdm(data[:250])):
    ft = np.fft.rfft(np.fft.ifftshift(x))
    f_MHz = np.fft.rfftfreq(ppifg, d=1e-9) * 1e-6
    apply_filter(f_MHz, ft, filt)
    x = np.fft.fftshift(np.fft.irfft(ft))
    x = np.roll(x, center - np.argmax(x))

    x_a = x[center - apod // 2 : center + apod // 2]
    ft_a = np.fft.rfft(np.fft.ifftshift(x_a))
    p_a = np.angle(ft_a)
    f_a = np.fft.rfftfreq(len(x_a), d=1e-9) * 1e-6

    f_c = 148
    f_ll, f_ul = 81, 240
    (ind,) = np.logical_and(f_ll < f_a, f_a < f_ul).nonzero()
    polyfit = np.polyfit((f_a - f_c)[ind], np.unwrap(p_a)[ind], deg=2)
    poly1d = np.poly1d(polyfit)

    ft *= np.exp(-1j * poly1d(f_MHz - f_c))
    x = np.fft.fftshift(np.fft.irfft(ft))

    _overwrite[n] = x
