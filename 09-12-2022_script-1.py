import numpy as np
import matplotlib.pyplot as plt
import phase_correction as pc
import digital_phase_correction as dpc
import scipy.constants as sc
import nyquist_bandwidths as nq

# ______________________________________________________________________________________________________________________
path = r'C:\Users\pchan\SynologyDrive\Research_Projects\Microscope\FreeRunningSpectra/'
name = 'First_Dual_Comb_From_Microscope_19152x51896.bin'
ppifg = 51896
center = ppifg // 2
N_ifgs = 19152 - 1
data = np.fromfile(path + name, '<h')
data = data[center:-center].copy()
data.resize((N_ifgs, ppifg))
data.resize(int(np.ceil(N_ifgs / 10)), 10, ppifg)
data = np.mean(data, 1)

# ______________________________________________________________________________________________________________________
# plt.figure()
# n = 10
# N = 10
# plt.plot(data[:N][:, center - n:center + n].T)

# ______________________________________________________________________________________________________________________
ll, ul = 0.3241, 0.4521
pdiff = dpc.get_pdiff(data, ll, ul, 200)
h = 0
step = 250
while h < len(data):
    dpc.apply_t0_and_phi0_shift(pdiff[h:h + step], data[h: h + step])
    h += step
    print(len(data) - h)

# ______________________________________________________________________________________________________________________
plt.figure()
[plt.plot(i[center - 50:center + 50]) for i in data[::10]]

avg = np.mean(data, 0)
spec = pc.fft(avg).__abs__()

# ______________________________________________________________________________________________________________________
frep = 1010e6 - 10017498.5
Nyquist_Window = 4

center = ppifg // 2
Nyq_Freq = center * frep
translation = (Nyquist_Window - 1) * Nyq_Freq
nu = np.linspace(0, Nyq_Freq, center) + translation
wl = np.where(nu > 0, sc.c * 1e6 / nu, np.nan)

if Nyquist_Window % 2 == 0:
    spec_final = spec[:center]  # negative frequency side
else:
    spec_final = spec[center:]  # positive frequency side

# ______________________________________________________________________________________________________________________
plt.figure()
plt.plot(wl, spec_final)

nyq = center * frep
dfr = nq.bandwidth(frep, nyq)

# ______________________________________________________________________________________________________________________
vi = sc.c / 3.85e-6
vf = sc.c / 2.9e-6
window_bounds = nq.find_allowed_nyquist_bandwidths(vi, vf)[::-1]
