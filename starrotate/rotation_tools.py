"""
A set of functions for measuring rotation periods and plotting the results.
"""


import os
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import kplr

plotpar = {'axes.labelsize': 25,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def transit_mask(self, t0, duration, porb):
    """
    Mask out transits

    Args:
        t0 (float): The reference time of transit. For Kepler data you may
            need to subtract 2454833.0 off this number.
        duration (float): The transit duration.
        porb (float): The planet's orbital period.
    """
    _t0 = float(t0) % porb
    _duration = float(duration) / 24.
    m = np.abs((self.time - _t0 + 0.5*self.porb) \
                % self.porb - 0.5*self.porb) < 1.5*_duration
    return m

def download_light_curves(KID, download_path, lcpath):
    client = kplr.API(data_root=download_path)
    star = client.star(str(int(KID)).zfill(9))
    star.get_light_curves(fetch=True, short_cadence=False);
    time, flux, flux_err = load_kepler_data(lcpath)
    return time, flux, flux_err


def load_kepler_data(LC_DIR):
    """
    Load data and join quarters together.
    Returns the time, flux and flux_err arrays.
    """

    # The names of the light curve fits files.
    fnames = sorted(glob.glob(os.path.join(LC_DIR, "*fits")))

    # load and median normalize the first quarter
    time, flux, flux_err = load_and_normalize(fnames[0])

    # Concatenate with the remaining median normalized quarters
    for fname in fnames[1:]:
        x, y, yerr = load_and_normalize(fname)
        time = np.concatenate((time, x))
        flux = np.concatenate((flux, y))
        flux_err = np.concatenate((flux_err, yerr))

    return time, flux, flux_err


def load_and_normalize(fname):
    """
    Load one quarter, remove bad points and median normalize it.
    """

    # Load the data and pull out the time and flux arrays.
    hdulist = fits.open(fname)
    t = hdulist[1].data
    time = t["TIME"]
    flux = t["PDCSAP_FLUX"]
    flux_err = t["PDCSAP_FLUX_ERR"]
    q = t["SAP_QUALITY"]

    # Mask out bad quality points and NaNs.
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
            (q == 0)
    x = time[m]

    # Median normalize
    med = np.median(flux[m])
    y = flux[m]/med - 1
    yerr = flux_err[m]/med

    return x, y, yerr


def dan_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    Args:
        x (array): The time series. If multidimensional, set the time axis
            using the ``axis`` keyword argument and the function will be
            computed for every other axis.
        axis (Optional[int]): The time axis of ``x``. Assumed to be the first
            axis if not specified.
        fast (Optional[bool]): If ``True``, only use the largest ``2^n``
            entries for efficiency. (default: False)

    Returns:
        acf (array): The acf array.
    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    m[axis] = 0
    return acf / acf[m]


def interp(x_gaps, y_gaps, interval):
    f = spi.interp1d(x_gaps, y_gaps, kind="zero")
    x = np.arange(x_gaps[0], x_gaps[-1], interval)
    return x, f(x)


def simple_acf(x_gaps, y_gaps, interval):
    """
    Compute an autocorrelation function and a period.

    Applies interpolation, smoothing and peak detection to estimate a
    rotation period.

    Args:
        x_gaps (array): The time array.
        y_gaps (array): The flux array.
        interval (Optional[float]): The time interval between successive
            observations. The default is Kepler cadence.

    Returns:
        lags (array): The array of lag times in days.
        acf (array): The autocorrelation function.
        period (float): The period estimated from the highest peak in the ACF.
    """

    # First of all: interpolate to an evenly spaced grid
    x, y = interp(x_gaps, y_gaps, interval)

    # fit and subtract straight line
    AT = np.vstack((x, np.ones_like(x)))
    ATA = np.dot(AT, AT.T)
    m, b = np.linalg.solve(ATA, np.dot(AT, y))
    y -= m*x + b

    # perform acf
    acf = dan_acf(y)

    # create 'lags' array
    lags = np.arange(len(acf))*interval

    N = len(acf)
    double_acf, double_lags = [np.zeros((2*N)) for i in range(2)]
    double_acf[:N], double_lags[:N] = acf[::-1], -lags[::-1]
    double_acf[N:], double_lags[N:] = acf, lags
    acf, lags = double_acf, double_lags

    # smooth with Gaussian kernel convolution
    Gaussian = lambda x, sig: 1./(2*np.pi*sig**.5) * np.exp(-0.5*(x**2)/
                                                            (sig**2))
    conv_func = Gaussian(np.arange(-28, 28, 1.), 9.)
    acf_smooth = np.convolve(acf, conv_func, mode='same')

    # just use the second bit (no reflection)
    acf_smooth, lags = acf_smooth[N:], lags[N:]

    # cut it in half
    m = lags < max(lags)/2.
    acf_smooth, lags = acf_smooth[m], lags[m]

    # ditch the first point
    acf_smooth, lags = acf_smooth[1:], lags[1:]

    # find all the peaks
    peaks = np.array([i for i in range(1, len(lags)-1)
                     if acf_smooth[i-1] < acf_smooth[i] and
                     acf_smooth[i+1] < acf_smooth[i]])


    # find the highest peak
    if len(peaks):
        m = acf_smooth == max(acf_smooth[peaks])
        highest_peak = acf_smooth[m][0]
        period = lags[m][0]
    else:
        period = 0.

    return lags, acf_smooth, period
