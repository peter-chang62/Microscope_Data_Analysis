# %%
import numpy as np
import clipboard as cr
import matplotlib.pyplot as plt
import pynlo
from scipy.constants import c
from scipy.integrate import simpson

# %% --------------------------------------------------------------------------
v_min = c / 3000e-9
v_max = c / 800e-9
v0 = c / 1560e-9
npts = 2**13
t_window = 20e-12
e_p = 4.0e-9
t_fwhm = 250e-15
pulse_grat = pynlo.light.Pulse.Sech(
    npts,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    t_window,
)

pulse_hnlf = pulse_grat.copy()
pulse_osc = pulse_grat.copy()
pulse_osc.e_p = 0.05e-9

# %% --------------------------------------------------------------------------
s_grat = np.genfromtxt("SPECTRUM_GRAT_PAIR.txt")
v_grid = c / (s_grat[:, 0] * 1e-9)
pulse_grat.import_p_v(v_grid, s_grat[:, 1])

s_hnlf = np.genfromtxt("Spectrum_Stitched_Together_wl_nm.txt")
v_grid = c / (s_hnlf[:, 0] * 1e-9)
pulse_hnlf.import_p_v(v_grid, s_hnlf[:, 1])

# pulse_hnlf.e_p = 2000e-9

# %% --------------------------------------------------------------------------
data = np.load("run.npy", mmap_mode="r")
p_v_mir = data[-1][100:].copy()
nu = np.fft.rfftfreq(77760, d=1e-9)[100:] * 77762
nu += nu[-1] * 2
wl = c * 1e6 / nu

ind = ~np.isnan(p_v_mir)

area = simpson(p_v_mir[ind], x=nu[ind])
power = area * 1e9  # W
factor = power / 3e-3
p_v_mir /= factor

# %% --------------------------------------------------------------------------
# norm = pulse_hnlf.p_v.max()
fact_pulse = pulse_grat.v_grid**2 / c * 1e12
fact_mir = nu**2 / c * 1e12
p_v_osc = pulse_osc.p_v * fact_pulse
p_v_grat = pulse_grat.p_v * fact_pulse
p_v_hnlf = pulse_hnlf.p_v * fact_pulse
p_v_mir_plot = p_v_mir * fact_mir

ymin, ymax = p_v_hnlf.max() * 1e-6, p_v_hnlf.max() * 5
(ind_osc,) = np.logical_and(ymin < p_v_osc, p_v_osc < ymax).nonzero()
(ind_grat,) = np.logical_and(ymin < p_v_grat, p_v_grat < ymax).nonzero()
(ind_hnlf,) = np.logical_and(ymin < p_v_hnlf, p_v_hnlf < ymax).nonzero()
(ind_mir,) = np.logical_and(ymin < p_v_mir_plot, p_v_mir_plot < ymax).nonzero()

fig, ax = plt.subplots(1, 1, figsize=np.array([5.51, 3.14]))
ax.semilogy(
    pulse_hnlf.wl_grid[ind_hnlf] * 1e6, p_v_hnlf[ind_hnlf], label="supercontinuum"
)
ax.semilogy(pulse_osc.wl_grid[ind_osc] * 1e6, p_v_osc[ind_osc], label="oscillator")
ax.semilogy(
    pulse_grat.wl_grid[ind_grat] * 1e6, p_v_grat[ind_grat], label="amplifier output"
)
ax.semilogy(wl[ind_mir], p_v_mir_plot[ind_mir], label="MIR")

alpha = 0.4
ax.fill_between(pulse_hnlf.wl_grid[ind_hnlf] * 1e6, p_v_hnlf[ind_hnlf], alpha=alpha)
ax.fill_between(pulse_osc.wl_grid[ind_osc] * 1e6, p_v_osc[ind_osc], alpha=alpha)
ax.fill_between(pulse_grat.wl_grid[ind_grat] * 1e6, p_v_grat[ind_grat], alpha=alpha)
ax.fill_between(wl[ind_mir], p_v_mir_plot[ind_mir], alpha=0.4)

ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("log PSD (mW / nm)")
ax.set_ylim(ymin=ymin)
ax.legend(loc="best")
fig.tight_layout()
