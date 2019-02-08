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
client = kplr.API()

plotpar = {'axes.labelsize': 25,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def download_light_curves(KID):
    star = client.star(KID)
    star.get_light_curves(fetch=True, short_cadence=False);
    time, flux, flux_err = load_kepler_data(
        "/Users/ruthangus/.kplr/data/lightcurves/{}".format(str(KID).
                                                            zfill(9)))
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


def transit_mask(time, t0, dur, porb):
    if t0 is None or dur is None or porb is None:
        return None

    # How many transits?
    ntransit = int((time[-1] - time[0])//porb)

    transit = (t0 - .5*dur < time) * (time < t0 + .5*dur)
    for i in range(ntransit):
        transit += (t0 + i*porb - .5*dur < time) * (time < t0 + i*porb + .5*dur)
    return transit


def dan_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.
    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.
    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)
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


def simple_acf(x_gaps, y_gaps, interval=0.02043365):

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
        print(period)
    else:
        period = 0.

    return lags, acf_smooth, period
