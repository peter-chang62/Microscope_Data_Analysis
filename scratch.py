# %%
import numpy as np
import matplotlib.pyplot as plt
import clipboard as cr
from scipy.constants import c


# %%
path = r"/Volumes/Peter SSD/Research_Projects/Microscope/Images/04-15-2023/bckgnd_stream_25704x77760.npy"
data = np.load(path, mmap_mode="r")
ppifg = 77760
center = ppifg // 2
shape = data.shape
data.resize(data.size)
data = data[center:-center]
data.resize((shape[0] - 1, shape[1]))


ft = np.fft.rfft(np.fft.fftshift(data[0]))
f = np.fft.rfftfreq(ppifg)

resolution = 100
N = ppifg // resolution
N = N if N % 2 == 0 else N + 1

ifg_a = data[0].copy()
ifg_a = ifg_a[center - N // 2 : center + N // 2]

ft_a = np.fft.rfft(np.fft.fftshift(ifg_a))
f_a = np.fft.rfftfreq(N)

# %%
lims = 0.0164, 0.2409
idx_f = np.logical_and(lims[0] < f, f < lims[1]).nonzero()
idx_a = np.logical_and(lims[0] < f_a, f_a < lims[1]).nonzero()
norm_f = abs(ft)[idx_f].max()
norm_a = abs(ft)[idx_a].max()

# %%
nu = np.fft.rfftfreq(ppifg, d=1e-9) * ppifg
nu += nu[-1] * 2
wl = c / nu

nu_a = np.fft.rfftfreq(N, d=1e-9) * ppifg
nu_a += nu_a[-1] * 2
wl_a = c / nu_a

# %%
plt.figure()
plt.plot(wl * 1e6, abs(ft) / norm_a, ".", markersize=1, label="1 GHz resolution")
plt.plot(wl_a * 1e6, abs(ft_a) / norm_a, label="100 GHz resolution")
plt.ylim(-0.05, 1.05)
plt.legend(loc="best")
plt.xlabel("wavelength ($\\mathrm{\\mu m}$)")
plt.tight_layout()
