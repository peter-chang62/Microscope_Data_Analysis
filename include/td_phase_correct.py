"""" optimiztion in the time domain doesn't work! It never gives an answer different from the initial guess,
even when it's obviously at an unstable maximum. """

import numpy as np
import scipy.signal as ss
import scipy.optimize as so


# shift a signal in time, I've vetted that this gives expected results
def shift(x, dt):
    ft = np.fft.fft(np.fft.ifftshift(x))
    freq = np.fft.fftfreq(len(ft))
    phase = np.exp(1j * 2 * np.pi * freq * dt)
    ft *= phase
    return np.fft.fftshift(np.fft.ifft(ft)).real


# apply a phase offset, I've also vetted that this cycles the ceo phase by the expected amount
def phi0_shift(x, offst):
    hbt = ss.hilbert(x)
    hbt *= np.exp(1j * offst)
    return hbt.real


# function to minimize, I've vetted that this at least looks physical
def error_dt_offst(X, x, x0, zoom=None):
    dt, phi0 = X
    ppifg = len(x)
    center = ppifg // 2
    y = shift(x, dt)  # shift x by dt
    y = phi0_shift(y, phi0)  # now apply phase offset to the shifted interferogram

    if zoom is not None:
        return np.mean((x0[center - zoom // 2: center + zoom // 2] -
                        y[center - zoom // 2: center + zoom // 2]) ** 2)
    else:
        return np.mean((x0 - y) ** 2)


# same as func but only with phase offset
# I've vetted that this at least looks physical
def error_offst(X, x, x0, zoom=None):
    phi0 = X
    ppifg = len(x)
    center = ppifg // 2
    y = phi0_shift(x, phi0)  # apply phase offset to x

    if zoom is not None:
        return np.mean((x0[center - zoom // 2: center + zoom // 2] -
                        y[center - zoom // 2: center + zoom // 2]) ** 2)
    else:
        return np.mean((x0 - y) ** 2)


class Optimize:
    def __init__(self, data):
        self.data = data
        self.data = (data.T / np.max(data, axis=1)).T
        self.data = (data.T - np.mean(data, axis=1)).T

    def error_shift_offst(self, X, n, zoom):
        return error_dt_offst(X, self.data[n], self.data[0], zoom)

    def error_offst(self, X, n, zoom):
        return error_offst(X, self.data[n], self.data[0], zoom)

    def phase_correct(self, zoom):
        offst_with_dt_guess = np.zeros((len(self.data), 2))
        CORR = self.data.copy()
        ERROR = np.zeros((len(self.data)))
        for n, i in enumerate(self.data):
            # dt, phi0 = X
            res = so.minimize(fun=self.error_shift_offst,
                              x0=np.array([0, 0]),
                              args=(n, zoom),
                              method='Nelder-Mead')
            offst_with_dt_guess[n] = res.x
            ERROR[n] = res.fun

            x = shift(CORR[n], res.x[0])
            x = phi0_shift(x, res.x[1])
            CORR[n] = x

            print(res.x)

        self.CORR = CORR
        self.ERROR = ERROR
