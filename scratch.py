import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt

path = r"/Users/peterchang/Resilio Sync/OvarianFTIR/"
data = np.fromfile(path + "D1", "<f")

data.shape = (394, 1280, 1280)
wnum = np.genfromtxt(path + "D1.hdr", skip_header=18, skip_footer=1, delimiter=",")
wl = 1e4 / wnum

dcs_bckgnd = abs(np.fft.rfft(np.fft.ifftshift(np.load("avg_off_bio_sample.npy"))))
dcs_bio = abs(np.fft.rfft(np.fft.ifftshift(np.load("avg_on_bio_sample.npy"))))
absrptn = dcs_bio / dcs_bckgnd
absrbnc = -np.log(absrptn)
v_grid = np.fft.rfftfreq(77760, d=1e-9) * 77760
v_grid += v_grid[-1] * 2
wl_grid = 299792458.0 / v_grid * 1e6

# %%
(ind_dcs,) = np.logical_and(3.03 < wl_grid, wl_grid < 3.65).nonzero()
(ind_reddy,) = np.logical_and(3.03 < wl, wl < 3.65).nonzero()
plt.figure()
plt.plot(wl[ind_reddy], data[:, 1280 // 2, 1280 // 2][ind_reddy])
plt.plot(wl_grid[ind_dcs], absrbnc[ind_dcs])
plt.ylim(0, .4)
plt.xlabel("wavelength ($\\mathrm{\\mu m}$)")
plt.tight_layout()
