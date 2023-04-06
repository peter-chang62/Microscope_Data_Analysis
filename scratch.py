import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt

# path = r"/Users/peterchang/Resilio Sync/OvarianFTIR/"
path = r"temp/OvarianFTIR/"
data = np.fromfile(path + "D1", "<f")

data.shape = (394, 1280, 1280)
wnum = np.genfromtxt(path + "D1.hdr", skip_header=18, skip_footer=1, delimiter=",")
wl = 1e4 / wnum

wnum_resolution = np.mean(np.diff(wnum))
c = 299792458
cm = 1e-2
freq_resolution = wnum_resolution * c / cm

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
fig, ax = plt.subplots(1, 1)
ax.plot(wl[ind_reddy], data[:, 1280 // 2, 1280 // 2][ind_reddy])
ax.plot(wl_grid[ind_dcs], absrbnc[ind_dcs])
conversion = lambda x: 1e4 / x
ax2 = ax.secondary_xaxis("top", functions=(conversion, conversion))
ax2.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax.set_ylim(0, 0.4)
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
fig.tight_layout()
