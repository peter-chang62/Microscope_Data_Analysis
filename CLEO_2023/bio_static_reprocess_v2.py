# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tables as tb
import clipboard
from scipy.constants import c


def rfft(x, axis=-1):
    return np.fft.rfft(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=-1):
    return np.fft.fftshift(np.fft.irfft(x, axis=axis), axes=axis)


# %% -----
path = (
    r"/Volumes/Peter SSD/Research_Projects/"
    + r"Microscope/FreeRunningSpectra/03-23-2023/"
)
ppifg = 77760
center = ppifg // 2

# %% ----- load data
# data = np.load(path + "think_off_bio_sample_51408x77760.npy", mmap_mode="r")
# shape = data.shape
# data.resize(data.size)
# data = data[center:-center]
# data.resize(shape[0] - 1, shape[1])


# %% ----- for any apodization to work, f0 needs to be removed first with
#   the full sampling
# nu = np.fft.rfftfreq(ppifg, d=1e-9)

# # fig, ax = plt.subplots(1, 1)
# # ft = rfft(data[0])
# # ax.plot(nu, ft.__abs__())

# filters = [
#     (249950503.54762402, 250059278.75096786),
#     (274375602.8401505, 288976135.6304068),
#     (332488660.9012865, 333949225.1273356),
#     (436713601.9544551, 437298771.8512813),
#     (0.0, 544123.6333513374),
#     (499743250.1701721, 500339365.011353),
# ]
# idx_filters = [np.logical_and(i[0] <= nu, nu <= i[1]).nonzero()[0] for i in filters]
# idx_filters = np.hstack(idx_filters)

# # filter out f0
# np.save("off_bio_sample_nof0.npy", data)  # initialize overwrite file
# data_nof0 = np.load("off_bio_sample_nof0.npy", "r+")
# for n, ifg in enumerate(tqdm(data)):
#     ft = rfft(ifg)
#     ft[idx_filters] = 0
#     data_nof0[n] = irfft(ft)

# # %%----- I still find this digital phase correction to work the best ...
# window = 100
# nu_window = np.fft.rfftfreq(window, d=1e-9)
# nu_full = np.fft.rfftfreq(ppifg, d=1e-9)
# data_nof0 = np.load("off_bio_sample_nof0.npy", "r")
# np.save("off_bio_sample_nof0_pc.npy", data_nof0)  # initialize overwrite file
# data_nof0_pc = np.load("off_bio_sample_nof0_pc.npy", "r+")

# for n, ifg in enumerate(tqdm(data_nof0)):
#     ft_window = rfft(ifg[center - window // 2 : center + window // 2])
#     p_window = np.unwrap(np.angle(ft_window))

#     # find fit range
#     # fig, ax = plt.subplots(1, 1)
#     # ax2 = ax.twinx()
#     # ax.plot(nu_window, abs(ft_window), ".")
#     # ax2.plot(nu_window, p_window, color="C1", marker=".")

#     filt = (64818548.38709678, 220060483.87096775)
#     (idx_filters,) = np.logical_and(filt[0] < nu_window, nu_window < filt[1]).nonzero()
#     nu0 = np.diff(filt) / 2 + filt[0]
#     polyfit = np.polyfit(nu_window[idx_filters] - nu0, p_window[idx_filters], deg=2)

#     p_subtract = np.poly1d(polyfit[-2:])(nu_full - nu0)
#     ft_full = rfft(data_nof0[n])
#     ft_full *= np.exp(-1j * p_subtract)

#     data_nof0_pc[n] = irfft(ft_full)

# # %% ----- calculate running average
# data_nof0_pc = np.load("off_bio_sample_nof0_pc.npy", "r")
# np.save(
#     "off_bio_sample_nof0_pc_running_avg.npy", data_nof0_pc
# )  # initialize overwrite file
# average = np.load("off_bio_sample_nof0_pc_running_avg.npy", "r+")
# avg = np.zeros(ppifg)
# for n, ifg in enumerate(tqdm(data_nof0_pc)):
#     avg = (avg * n + ifg) / (n + 1)
#     average[n] = avg

# %% ----- 2D map for SNR
# filt = (102988308.86410272, 173658075.50593784)
# average = np.load("off_bio_sample_nof0_pc_running_avg.npy", "r")

# resolution_grid = np.arange(1, 101)
# SNR = np.zeros((resolution_grid.size, average.shape[0]))
# for n, resolution in enumerate(tqdm(resolution_grid)):
#     if resolution == 1:
#         nu = np.fft.rfftfreq(ppifg, d=1e-9)
#         (idx_filt,) = np.logical_and(filt[0] < nu, nu < filt[1]).nonzero()

#         ft_ref = rfft(average[-1])[idx_filt[0] : idx_filt[-1]]

#         snr = np.zeros(average.shape[0])
#         for m, ifg in enumerate(tqdm(average)):
#             ft = rfft(ifg)[idx_filt[0] : idx_filt[-1]]
#             snr[m] = np.std(np.log(ft / ft_ref))
#         SNR[n] = snr

#     else:
#         window = ppifg // resolution
#         window = window if window % 2 == 0 else window + 1
#         nu = np.fft.rfftfreq(window, d=1e-9)
#         (idx_filt,) = np.logical_and(filt[0] < nu, nu < filt[1]).nonzero()

#         ifg_ref = average[-1][center - window // 2 : center + window // 2]
#         ft_ref = rfft(ifg_ref)[idx_filt[0] : idx_filt[-1]].__abs__()

#         snr = np.zeros(average.shape[0])
#         for m, ifg in enumerate(tqdm(average)):
#             ifg = ifg[center - window // 2 : center + window // 2]
#             ft = rfft(ifg)[idx_filt[0] : idx_filt[-1]].__abs__()
#             snr[m] = np.std(np.log(ft / ft_ref))
#         SNR[n] = snr

# np.save("off_bio_sample_snr_avg_res_1ghz_step.npy", SNR)

# %% ----- look at 2D map for SNR
data = np.load("off_bio_sample_nof0_pc_running_avg.npy", mmap_mode="r")
sigma = np.load("off_bio_sample_snr_avg_res_1ghz_step.npy")

sigma = np.where(sigma == 0, np.nan, sigma)
factor_improvement_res = np.nanmean(sigma[0] / sigma, axis=1)
factor_improvement_t = np.nanmean(sigma.T[0] / sigma.T, axis=1)

t_log = np.log(np.arange(factor_improvement_t.size) + 1)
n_log = np.log(np.arange(factor_improvement_res.size) + 1)
factor_improvement_res_log = np.log(factor_improvement_res)
factor_improvement_t_log = np.log(factor_improvement_t)

# %% ----- plotting
resolution = np.logspace(np.log10(1), np.log10(100), num=100)
apod = ppifg // resolution
apod[apod % 2 == 1] += 1
apod = apod.astype(int)
resolution = ppifg / apod
n = np.arange(sigma.shape[1]) + 1
tau = ppifg / 1e9

figsize = np.array([4.64, 3.63])
fig_r, ax_r = plt.subplots(1, 1, figsize=figsize)
img = ax_r.pcolormesh(
    n[0 : int(1e3)] * tau,
    resolution,
    1 / sigma[:, 0 : int(1e3)],
    cmap="jet",
)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_xlabel("time (s)")
ax_r_2 = ax_r.secondary_xaxis("top", functions=(lambda x: x / tau, lambda x: x * tau))
ax_r_2.set_xlabel("# of averaged spectra")
ax_r.set_ylabel("resolution (GHz)")
colorbar = plt.colorbar(img, label="SNR")
fig_r.tight_layout()

markersize = 4
fig_allan, ax_allan = plt.subplots(1, 1, figsize=figsize)
ax_allan.loglog(
    n[0 : int(1e3)] * tau,
    1 / sigma[0, 0 : int(1e3)],
    "o",
    linewidth=3,
    label="1 GHz res",
    markersize=markersize,
)
ax_allan.loglog(
    n[0 : int(1e3)] * tau,
    1 / sigma[1, 0 : int(1e3)],
    "o",
    linewidth=3,
    label="2 GHz res",
    markersize=markersize,
)
ax_allan.loglog(
    n[0 : int(1e3)] * tau,
    1 / sigma[9, 0 : int(1e3)],
    "o",
    linewidth=3,
    label="10 GHz res",
    markersize=markersize,
)
ax_allan.loglog(
    n[0 : int(1e3)] * tau,
    1 / sigma[99, 0 : int(1e3)],
    "o",
    linewidth=3,
    label="100 GHz res",
    markersize=markersize,
)
ax_allan_2 = ax_allan.secondary_xaxis(
    "top", functions=(lambda x: x / tau, lambda x: x * tau)
)
ax_allan.set_xlabel("time (s)")
ax_allan_2.set_xlabel("# of averaged spectra")
ax_allan.set_ylabel("SNR")
ax_allan.legend(loc="best")
fig_allan.tight_layout()

fig_allan_res, ax_allan_res = plt.subplots(1, 1, figsize=figsize)
nu_res = np.arange(sigma.shape[0]) + 1
ax_allan_res.loglog(
    nu_res, 1 / sigma[:, 10], "o", markersize=markersize, label=f"10 averages"
)
ax_allan_res.loglog(
    nu_res, 1 / sigma[:, 100], "o", markersize=markersize, label=f"100 averages"
)
ax_allan_res.loglog(
    nu_res, 1 / sigma[:, 1000], "o", markersize=markersize, label=f"1,000 averages"
)
ax_allan_res.loglog(
    nu_res, 1 / sigma[:, 10000], "o", markersize=markersize, label=f"10,000 averages"
)
# ax_allan_res.loglog(nu_res, 1 / sigma[:, 10], "o", markersize=markersize, label=f"10 ({np.round(10 * tau * 1e3)} ms) averages")
# ax_allan_res.loglog(nu_res, 1 / sigma[:, 100], "o", markersize=markersize, label=f"100 ({np.round(100 * tau * 1e3)} ms) averages")
# ax_allan_res.loglog(nu_res, 1 / sigma[:, 1000], "o", markersize=markersize, label=f"1,000 ({np.round(1000 * tau * 1e3)} ms) averages")
# ax_allan_res.loglog(nu_res, 1 / sigma[:, 10000], "o", markersize=markersize, label=f"10,000 ({np.round(10000 * tau * 1e3)} ms) averages")
ax_allan_2_res = ax_allan_res.secondary_xaxis(
    "top", functions=(lambda x: x * 1e9 / c / 100, lambda x: x * c * 100 / 1e9)
)
ax_allan_res.set_xlabel("resolution (GHz)")
ax_allan_2_res.set_xlabel("wavenumber ($\\mathrm{cm^{-1}}$)")
ax_allan_res.set_ylabel("SNR")
ax_allan_res.legend(loc="best")
fig_allan_res.tight_layout()

fig_impr, ax_impr = plt.subplots(2, 1, figsize=figsize)
ax_impr_t, ax_impr_n = ax_impr

# fig_impr_t, ax_impr_t = plt.subplots(1, 1)
n_ifg = np.arange(factor_improvement_t.size, dtype=float) + 1
t = n_ifg * ppifg / 1e9
ax_impr_t.loglog(t[:1000], factor_improvement_t[:1000], "o", markersize=markersize)
ax_impr_t.loglog(
    t[:1000], factor_improvement_t[0] * np.sqrt(n_ifg[:1000]), linewidth=2, linestyle="--"
)
ax_impr_t.set_xlabel("time (s)")
ax_impr_t.set_ylabel("$\\mathrm{SNR_\\tau \\; / \\; SNR_{78\\mu s}}$")

# fig_impr_n, ax_impr_n = plt.subplots(1, 1)
n = np.arange(factor_improvement_res.size) + 1
ax_impr_n.loglog(n, factor_improvement_res, "o", markersize=markersize)
ax_impr_n.loglog(n, factor_improvement_res[0] * np.sqrt(n), linewidth=2, linestyle="--")
ax_impr_n.set_xlabel("resolution (GHz)")
ax_impr_n.set_ylabel("$\\mathrm{SNR_{\\nu_{res}} \\; / \\; SNR_{1 GHz}}$")
fig_impr.tight_layout()
fig_impr.tight_layout()
fig_impr.tight_layout()


# %% ----- more plotting
dcs = lambda fr, dfr: fr**2 / (2 * dfr)


def N_bins(fr, dfr):
    dnu = dcs(fr, dfr)
    N = dnu / fr
    return N


dnu = np.linspace(5e12, 100e12, 5000)
dfrep_1ghz = dcs(1e9, dnu)
N_bins_1ghz = dnu / 1e9
dfrep_100MHz = dcs(100e6, dnu)
N_bins_100MHz = dnu / 100e6
dfrep_200MHz = dcs(200e6, dnu)
N_bins_200MHz = dnu / 200e6

fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.semilogy(dnu * 1e-12, dfrep_100MHz / N_bins_100MHz, label="100 MHz", linewidth=3)
ax.semilogy(dnu * 1e-12, dfrep_200MHz / N_bins_200MHz, label="200 MHz", linewidth=3)
ax.semilogy(dnu * 1e-12, dfrep_1ghz / N_bins_1ghz, label="1 GHz", linewidth=3)
ax.set_xlabel("bandwidth $\\mathrm{\\Delta \\nu}$ (THz)")
ax.set_ylabel("$\\mathrm{\\Delta f_r \\; / \\; N_0}$")
ax.legend(loc="best")
fig.tight_layout()
fig.tight_layout()
fig.tight_layout()
