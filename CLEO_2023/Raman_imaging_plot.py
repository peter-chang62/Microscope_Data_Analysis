# %% -----
import numpy as np
import matplotlib.pyplot as plt
import clipboard as cr
import collections
from scipy.constants import c

pub = collections.namedtuple(
    "Publication",
    [
        "method",
        "spec_acq_spd",
        "pix_acq_spd",
        "bandwidth",
        "spec_res",
    ],
)


# %% --------------------------------------------------------------------------
def loglog(x, y, ax, color, label=None):
    if isinstance(x, np.ndarray):
        assert x.shape == y.shape, "x and y must have the same shape"
        ax.loglog(x, y, "-", linewidth=5, color=color, label=label, alpha=0.6)
    else:
        ax.loglog(x, y, "o", color=color, label=label)


# %% ----------------- Raman --------------------------------------------------
kee = pub("b_CARS", 59.0, 59.0, 2500.0, 13.0)
ploetz = pub("fSRM", 5e2, 1.0, 3400.0, 25.0)
evans = pub("CARS", 6.26e6, 6.25e6, 3.0, 3.0)
saar = pub("SRS", 1e7, 1e7, 3.0, 3.0)
fu_am_chem = pub("m-SRS", 5e3, 5e3, 100.0, 33.0)
liao = pub("m-SRS", 3.13e4, 3.13e4, 180, 11.25)
chowdary = pub("NIVI", 1e3, 1e2, 300, 0.9)
ozeki = pub(
    "swept-source-SRS",
    np.linspace(7.2e6, 7.2e4),
    np.linspace(7.2e6, 7.2e4),
    np.linspace(3.0, 300.0),
    3.0,
)
fu_phys_chem = pub(
    "spectral-focusing-SRS",
    np.linspace(2.34e5, 2.34e3),
    np.linspace(2.34e5, 2.34e3),
    np.linspace(10.0, 300.0),
    10.0,
)
napoli = pub(
    "spectral-focusing-CARS",
    np.linspace(1e5, 1e3),
    np.linspace(1e5, 1e3),
    np.linspace(10.0, 100.0),
    10.0,
)
lin = pub("spectral-focusing-SRS", 5.5e4, 5.5e4, 200.0, 10.0)
ideguchi = pub("comb-CARS", 100.0, 50.0, 1200.0, 10.0)

# %% ----------------- FTIR ---------------------------------------------------
nasse = pub("ftir_multbeam_fpa_sync", 2.65e3, 2.65e3, 3000, "unknown?")

# %% ----------------- DCS EOM ------------------------------------------------
khan = pub("dcs_fpa_eom_dfg_wvgd", 8.19e4, 8.19e4, 5.4, 0.6)

# %% ----------------- DCS mode-locked lasers ---------------------------------
# ghz = pub("GHz_MIR_DCS", 1.29e4, 25.72, 1000, 3.3)
ghz = pub("GHz_MIR_DCS", 1.29e4, 25.72, 1297, 3.3)  # set to match nyquist

# %% ----------------- QCL ------------------------------------------------
yeh = pub("QCL_DF-IR", 5461, 5461, 1100, 4)

# %% ----------------- scanless up-conversion camera imaging ------------------
zhao = pub("scanless_upconversion", 38.4e3, 38.4e3, 3015 - 640, 3.15)

# %% -------------- plot with pixel acquisition speed -------------------------
# fig, ax = plt.subplots(1, 1, figsize=np.array([8.3, 5.75]))
# ax.spines.top.set_visible(False)
# ax.spines.right.set_visible(False)

# # Raman
# loglog(kee.pix_acq_spd, kee.bandwidth, ax, color="C0", label="b-CARS")
# loglog(ploetz.pix_acq_spd, ploetz.bandwidth, ax, color="C1", label="f-SRM")
# loglog(evans.pix_acq_spd, evans.bandwidth, ax, color="C2", label="CARS")
# loglog(saar.pix_acq_spd, saar.bandwidth, ax, color="C4", label="SRS")
# loglog(fu_am_chem.pix_acq_spd, fu_am_chem.bandwidth, ax, color="C5", label="m-SRS")
# loglog(liao.pix_acq_spd, liao.bandwidth, ax, color="C5")
# loglog(chowdary.pix_acq_spd, kee.bandwidth, ax, color="C6", label="NIVI")
# loglog(
#     ozeki.pix_acq_spd,
#     ozeki.bandwidth,
#     ax,
#     color="C7",
#     label="swept-source-SRS",
# )
# loglog(
#     fu_phys_chem.pix_acq_spd,
#     fu_phys_chem.bandwidth,
#     ax,
#     color="C8",
#     label="spectral-focusing-SRS",
# )
# loglog(lin.pix_acq_spd, lin.bandwidth, ax, color="C8")
# loglog(
#     napoli.pix_acq_spd,
#     napoli.bandwidth,
#     ax,
#     color="C9",
#     label="spectral-focusing-CARS",
# )
# loglog(
#     ideguchi.pix_acq_spd,
#     ideguchi.bandwidth,
#     ax,
#     color="gold",
#     label="comb-CARS",
# )

# # FTIR
# loglog(
#     nasse.pix_acq_spd,
#     nasse.bandwidth,
#     ax,
#     color="darkslateblue",
#     label="multi-beam synchrotron FTIR",
# )

# # MIR DCS
# loglog(khan.pix_acq_spd, khan.bandwidth, ax, color="olive", label="MIR DCS EOM DFG")
# loglog(ghz.pix_acq_spd, ghz.bandwidth, ax, color="crimson", label="1 GHz MIR DCS")

# ax.set_xlabel("pixel acquisition speed (Hz)")
# ax.set_ylabel("optical bandwidth ($\\mathrm{cm^{-1}}$)")
# ax.legend(loc="best")
# fig.suptitle("bandwidth vs. pixel acquisition speed")
# fig.tight_layout()

# %% -------------- plot with spectral acquisition speed ----------------------
y_lim = (2.11058754816463, 4832.777493106092)
x_lim = (32.31428955730705, 18258176.431627173)
figsize = np.array([8.3, 5.75])

fig, ax = plt.subplots(
    1,
    1,
    figsize=figsize,
    num="bandwidth vs. spectral acquisition speed",
)
ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)

# Raman
loglog(kee.spec_acq_spd, kee.bandwidth, ax, color="C0", label="b-CARS")
loglog(ploetz.spec_acq_spd, ploetz.bandwidth, ax, color="C1", label="f-SRM")
loglog(evans.spec_acq_spd, evans.bandwidth, ax, color="C2", label="CARS")
loglog(saar.spec_acq_spd, saar.bandwidth, ax, color="C4", label="SRS")
loglog(fu_am_chem.spec_acq_spd, fu_am_chem.bandwidth, ax, color="C5", label="m-SRS")
loglog(liao.spec_acq_spd, liao.bandwidth, ax, color="C5")
loglog(chowdary.spec_acq_spd, kee.bandwidth, ax, color="C6", label="NIVI")
loglog(
    ozeki.spec_acq_spd,
    ozeki.bandwidth,
    ax,
    color="C7",
    label="swept-source-SRS",
)
loglog(
    fu_phys_chem.spec_acq_spd,
    fu_phys_chem.bandwidth,
    ax,
    color="C8",
    label="spectral-focusing-SRS",
)
loglog(lin.spec_acq_spd, lin.bandwidth, ax, color="C8")
loglog(
    napoli.spec_acq_spd,
    napoli.bandwidth,
    ax,
    color="C9",
    label="spectral-focusing-CARS",
)
loglog(
    ideguchi.spec_acq_spd,
    ideguchi.bandwidth,
    ax,
    color="gold",
    label="comb-CARS",
)

# FTIR
loglog(
    nasse.spec_acq_spd,
    nasse.bandwidth,
    ax,
    color="darkslateblue",
    label="multi-beam synchrotron FTIR",
)

# QCL
loglog(yeh.spec_acq_spd, yeh.bandwidth, ax, color="r", label="QCL DF-IR")

# MIR DCS
loglog(khan.spec_acq_spd, khan.bandwidth, ax, color="olive", label="EOM MIR DCS")
loglog(ghz.spec_acq_spd, ghz.bandwidth, ax, color="crimson", label="1 GHz MIR DCS")

# scanless upconversion
loglog(zhao.spec_acq_spd, zhao.bandwidth, ax, color="k", label="scanless up-conversion")

# Diagonal line
x_pt = np.array(x_lim)  # 1s -> video rate
y_pt = 1e9**2 / (2 * x_pt) / c / 100  # slope for GHz
log_stp = 1.0
N = 3
for i in np.arange(-log_stp * N, log_stp * (N + 1), log_stp):
    ax.plot(x_pt, y_pt * 10**i, linestyle="--", color="gray", alpha=0.5)

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_xlabel("spectral acquisition speed (Hz)")
ax.set_ylabel("optical bandwidth ($\\mathrm{cm^{-1}}$)")
# ax.legend(loc="best")
fig.tight_layout()

# %% ----- background colormap
fig, ax = plt.subplots(1, 1, figsize=figsize)
_x_bckgnd = np.linspace(*x_lim, 256)
_y_bckgnd = np.linspace(*y_lim, 256)
x_bckgnd, y_bckgnd = np.meshgrid(_x_bckgnd, _y_bckgnd)
z_bckgnd = x_bckgnd * y_bckgnd
ax.pcolormesh(x_bckgnd, y_bckgnd, z_bckgnd, cmap="GnBu")

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_xlabel("spectral acquisition speed (Hz)")
ax.set_ylabel("optical bandwidth ($\\mathrm{cm^{-1}}$)")
fig.tight_layout()

ax.axis(False)
