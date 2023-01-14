import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

try:
    import mkl_fft  # a faster fft
except:
    mkl_fft = np.fft  # otherwise just use numpy's fft


def fft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def normalize(vec):
    return vec / np.max(abs(vec))


# useful for plotting in order to determine good apodization window
# and frequency window to fit spectral phase
def get_phase(dat, N_apod, plot=True, ax=None, ax2=None):
    ppifg = len(dat)
    center = ppifg // 2
    ft = fft(dat[center - N_apod // 2: center + N_apod // 2])
    phase = np.unwrap(np.arctan2(ft.imag, ft.real))
    freq = np.fft.fftshift(np.fft.fftfreq(len(phase)))

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if ax2 is None:
            ax2 = ax.twinx()
        ax.plot(freq, phase, '.-')
        ax2.plot(freq, ft.__abs__(), '.-', color='k')
    return freq, phase, ft.__abs__(), ax, ax2


# modifies the ft array in place
def apply_t0_shift(pdiff, freq, ft):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    ft[:] *= np.exp(1j * freq * pdiff[:, 0][:, np.newaxis])


# modifies the hbt array in place
def apply_phi0_shift(pdiff, hbt):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    hbt[:] *= np.exp(1j * pdiff[:, 1][:, np.newaxis])


def get_pdiff(data, ll_freq, ul_freq, Nzoom=200):
    """
    :param data: 2D array of IFG's, row column order
    :param ppifg: int, length of each interferogram
    :param ll_freq: lower frequency limit for spectral phase fit, given on -.5 to .5 scale
    :param ul_freq: upper frequency limit for spectral phase fit, given on -.5 to .5 scale
    :param Nzoom: the apodization window for the IFG, don't worry about f0 since you are fitting the spectral phase,
    not doing a cross-correlation, you need to apodize or else your SNR isn't good enough to have a good fit, so
    plot it first before specifying this parameter, generally 200 is pretty good
    :return: pdiff, polynomial coefficients, higher order first
    """

    center = len(data[0]) // 2
    zoom = data[:, center - Nzoom // 2:center + Nzoom // 2]
    zoom = (zoom.T - np.mean(zoom, 1)).T

    # not fftshifted
    ft = fft(zoom, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(len(ft[0])))
    ll, ul = np.argmin(abs(freq - ll_freq)), np.argmin(abs(freq - ul_freq))

    phase = np.unwrap(np.arctan2(ft.imag, ft.real))
    phase = phase.T  # column order for polynomial fitting
    p = np.polyfit(freq[ll:ul], phase[ll:ul], 1).T
    pdiff = p[0] - p

    return pdiff


# modifies the data array in place
def apply_t0_and_phi0_shift(pdiff, data):
    freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
    ft = fft(data, 1)
    apply_t0_shift(pdiff, freq, ft)
    td = ifft(ft, 1).real

    # td is the linear phase corrected time domain data
    hbt = ss.hilbert(td)  # take hilbert transform of the linear phase corrected time domain data
    apply_phi0_shift(pdiff, hbt)  # multiply by constant phase offset
    hbt = hbt.real  # just take the real (no inverse)

    data[:] = hbt.real
