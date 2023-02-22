import numpy as np
import scipy.signal as ss
import scipy.optimize as so
import warnings
from tqdm import tqdm


# shift a signal in time, I've vetted that this gives expected results
def shift(x, dt):
    ft = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x))
    phase = np.exp(1j * 2 * np.pi * freq * dt)
    ft *= phase
    return np.fft.irfft(ft).real


# apply a phase offset, I've also vetted that this cycles the ceo phase by
# the expected amount
def phi0_shift(x, offst):
    hbt = ss.hilbert(x)
    hbt *= np.exp(1j * offst)
    return hbt.real


# function to minimize, I've vetted that this at least looks physical
def error_dt_offst(X, x, x0):
    dt, phi0 = X
    y = shift(x, dt)  # shift x by dt
    y = phi0_shift(y, phi0)  # now apply phase offset to the shifted ifg

    return np.mean((x0 - y) ** 2)


# same as func but only with phase offset
# I've vetted that this at least looks physical
def error_offst(X, x, x0):
    phi0 = X
    y = phi0_shift(x, phi0)  # apply phase offset to x

    return np.mean((x0 - y) ** 2)


class Optimize:
    def __init__(self, data):
        self.data = data

        # sometimes you pass a portion of the data zoomed into the
        # centerburst in which case you cannot afford to alter just a
        # portion of the data !!!
        # if data.size < 2e9:  # if the data size is less than 2 gigabytes
        #     self.data = (data.T - np.mean(data, axis=1)).T  # remove DC offset
        #     self.data = (data.T / np.max(data, axis=1)).T  # normalize
        # else:
        #     warnings.warn(
        #         "The DC offset was not removed due to the size of the file")

    def error_shift_offst(self, X, n):
        return error_dt_offst(X, self.data[n], self.data[0])

    def error_offst(self, X, n):
        return error_offst(X, self.data[n], self.data[0])

    def phase_correct(self,
                      data_to_shift=None,
                      overwrite_data_to_shift=True,
                      start_index=0,
                      end_index=None,
                      method='Powell'):
        # I've noticed Powell has worked best
        self.error = np.zeros((len(self.data)))

        if end_index is None:
            end_index = len(self.data)

        self.avg = 0

        h = 0
        for n in tqdm(range(start_index, end_index)):
            # dt, phi0 = X
            res = so.minimize(fun=self.error_shift_offst,
                              x0=np.array([0, 0]),
                              args=(n,),
                              method=method)
            self.error[n] = res.fun

            x = shift(data_to_shift[n], res.x[0])
            x = phi0_shift(x, res.x[1])
            if overwrite_data_to_shift:
                data_to_shift[n] = x

            self.avg = (self.avg * n + x) / (n + 1)

            # print(res.x, end_index - start_index - h - 1) # using tqdm now
            h += 1
