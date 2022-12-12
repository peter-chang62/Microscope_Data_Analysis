"""" optimiztion in the time domain doesn't work! It never gives an answer different from the initial guess,
even when it's obviously at an unstable maximum. """

import numpy as np
import phase_correction as pc
import scipy.signal as ss
import scipy.optimize as so
import matplotlib.pyplot as plt

zoom = 1500


# shift a signal in time, I've vetted that this gives expected results
def shift(x, dt):
    ft = np.fft.fft(np.fft.ifftshift(x))
    freq = np.fft.fftfreq(len(ft))
    phase = np.exp(1j * 2 * np.pi * freq * dt)
    ft *= phase
    return np.fft.fftshift(np.fft.ifft(ft)).real


# apply a phase offset, I've also vetted that this cycles the ceo phase by the expected amount
def apply_phi0_shift(x, offst):
    hbt = ss.hilbert(x)
    hbt *= np.exp(1j * offst)
    return hbt.real


# function to minimize, I've vetted that this at least looks physical
def error_dt_offst(X, x, x0):
    dt, phi0 = X

    y = shift(x, dt)  # shift x by dt
    y = apply_phi0_shift(y, phi0)  # now apply phase offset to the shifted interferogram
    return np.mean((x0[center - zoom // 2: center + zoom // 2] -
                    y[center - zoom // 2: center + zoom // 2]) ** 2)


# same as func but only with phase offset
# I've vetted that this at least looks physical
def error_offst(X, x, x0):
    phi0 = X

    y = apply_phi0_shift(x, phi0)  # apply phase offset to x
    return np.mean((x0[center - zoom // 2: center + zoom // 2] -
                    y[center - zoom // 2: center + zoom // 2]) ** 2)


class Optimize:
    def __init__(self, data):
        self.data = data

    def error_shift_offst(self, X, n):
        return error_dt_offst(X, self.data[n], self.data[0])

    def error_offst(self, X, n):
        return error_offst(X, self.data[n], self.data[0])


# ____________________________________ load the data ___________________________________________________________________
path = r"data/Dans_interferograms/"
data = np.genfromtxt(path + "Data.txt")
T = np.genfromtxt(path + "t_axis.txt")
N = np.arange(-len(data[0]) // 2, len(data[0]) // 2)
ppifg = len(N)
center = ppifg // 2

data = (data.T / np.max(data, axis=1)).T
data = (data.T - np.mean(data, axis=1)).T

# ______________________________ calculate time shifts from cross correlation __________________________________________
corr, delta_t = pc.t0_correct_via_cross_corr(data, zoom, False)

# __________________________________________ optimization ______________________________________________________________
optimize = Optimize(data)
offst_with_dt_guess = np.zeros((len(data), 2))
CORR = data.copy()
ERROR = np.zeros((len(data)))
for n, i in enumerate(data):
    # dt, phi0 = X
    res = so.minimize(fun=optimize.error_shift_offst,
                      # x0 = np.array([delta_t[n], 0]),
                      x0=np.array([0, 0]),
                      args=(n,),
                      method='Nelder-Mead')
    offst_with_dt_guess[n] = res.x
    ERROR[n] = res.fun

    x = shift(CORR[n], res.x[0])
    x = apply_phi0_shift(x, res.x[1])
    CORR[n] = x

    print(res.x)

# __________________________________________ optimization ______________________________________________________________
# optimize = Optimize(data)
# offst_no_dt_guess = np.zeros((len(data), 2))
# for n, i in enumerate(data):
#     res = so.minimize(optimize.error_offst,
#                       np.array([0]),
#                       (n,),
#                       method='Powell')
#     offst_no_dt_guess[n] = res.x
#     print(res.x)

# __________________________________________ some diagnostics __________________________________________________________
# # this is odd, it should have found a few of the optima... func2 is sooooo well behaved
# # looks like switching to Powell's method did the trick!
# Phi = np.linspace(-2 * np.pi, 2 * np.pi, 500)
# F = np.zeros((len(data), len(Phi)))
#
# optimize = Optimize(data)
# for n, i in enumerate(optimize.data):
#     f = np.zeros(len(Phi))
#
#     for m, phi in enumerate(Phi):
#         f[m] = optimize.error_offst(phi, n)
#
#     F[n] = f
#     print(n)
