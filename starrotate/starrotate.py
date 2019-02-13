"""
A script for measuring the rotation periods of a set of stars.
"""

import numpy as np
from .rotation_tools import simple_acf
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import pandas as pd

plotpar = {'axes.labelsize': 25,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20}
           #'text.usetex': True}
plt.rcParams.update(plotpar)


class RotationModel(object):

    def __init__(self, time, flux, flux_err, t0=None,
                 duration=None, porb=None):

        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.t0, self.duration, self.porb = t0, duration, porb

    def plot_lc_zoom(self, zoom=100):
        plt.figure(figsize=(20, 5))
        plt.plot(self.raw_time, self.raw_flux, ".", color="C1", ms=2,
                    zorder=0);
        plt.xlabel("Time [days]")
        plt.ylabel("Relative Flux");
        plt.xlim(self.time[0], self.time[0] + zoom)
        plt.subplots_adjust(bottom=.15)

    def plot_lc(self):
        # Plot the light curve
        plt.figure(figsize=(20, 5))
        plt.plot(self.raw_time, self.raw_flux, "k.", ms=.5);
        plt.xlabel("Time [days]")
        plt.ylabel("Relative Flux");
        plt.subplots_adjust(bottom=.2)

    def plot_transit_mask(self):
        ms = 5
        plt.figure(figsize=(20, 5))
        plt.plot(self.raw_time, self.raw_flux, ".", color="C1", ms=ms,
                    zorder=0);
        plt.plot(self.time, self.flux, ".", color="C0", ms=ms, zorder=1);
        plt.xlabel("Time [days]")
        plt.ylabel("Relative Flux");
        plt.subplots_adjust(bottom=.2)

    def transit_mask(self, _t0, _duration, porb):
        t0 = float(_t0) % porb
        duration = float(_duration) / 24.

        m = np.abs((self.raw_time - t0 + 0.5*self.porb) \
                   % self.porb - 0.5*self.porb) < 1.5*duration
        return m

    def LS_rotation(self):

        assert len(self.flux) == sum(np.isfinite(self.flux)), "Remove NaNs" \
            " from your flux array before trying to compute a periodogram."

        # Calculate a LS period
        results = xo.estimators.lomb_scargle_estimator(
            self.time, self.flux, max_peaks=1, min_period=1.0,
            max_period=30.0, samples_per_peak=50)

        self.freq, self.power = results["periodogram"]
        peak = results["peaks"][0]
        self.ls_period = peak["period"]
        return peak["period"]

    def pgram_plot(self):
        # Make a plot of the periodogram.
        plt.figure(figsize=(16, 9))
        plt.plot(-np.log10(self.freq), self.power, "k", zorder=0)
        plt.axvline(np.log10(self.ls_period), color="C1", lw=4, alpha=0.5,
                    zorder=1)
        plt.xlim((-np.log10(self.freq)).min(), (-np.log10(self.freq)).max())
        plt.yticks([])
        plt.xlabel("log10(Period [days])")
        plt.ylabel("Power");
        plt.subplots_adjust(left=.15, bottom=.15)

    def ACF_rotation(self):
        lags, acf, acf_period = simple_acf(self.time, self.flux)

        self.lags = lags
        self.acf = acf
        self.acf_period = acf_period
        return acf_period

    def acf_plot(self):
        plt.figure(figsize=(16, 9))
        plt.plot(self.lags, self.acf, "k")
        plt.axvline(self.acf_period, color="C1")
        plt.xlabel("Period [days]")
        plt.ylabel("Correlation")
        plt.xlim(0, max(self.lags))
        plt.subplots_adjust(left=.15, bottom=.15)

    def GP_rotation(self, init_period=None):
        x = self.time
        y = self.flux
        yerr = self.flux_err

        if init_period is None:
            # Calculate ls period
            init_period = self.LS_rotation()

        with pm.Model() as model:

            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sd=10.0)

            # A jitter term describing excess white noise
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(yerr)), sd=5.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
            logperiod = pm.Normal("logperiod", mu=np.log(init_period),
                                  sd=5.0)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Track the period as a deterministic
            period = pm.Deterministic("period", tt.exp(logperiod))

            # Set up the Gaussian Process model
            kernel = xo.gp.terms.RotationTerm(
                log_amp=logamp,
                period=period,
                log_Q0=logQ0,
                log_deltaQ=logdeltaQ,
                mix=mix
            )
            gp = xo.gp.GP(kernel, x, yerr**2 + tt.exp(logs2), J=4)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(y - mean))

            # Compute the mean model prediction for plotting purposes
            # pm.Deterministic("pred", gp.predict())

            # Optimize to find the maximum a posteriori parameters
            self.map_soln = pm.find_MAP(start=model.test_point)

            # Plot the MAP fit.
            # if self.plot:
            #     plt.figure(figsize=(20, 5))
            #     # plt.plot(x, y, "k.", ms=.5, label="$\mathrm{data}$")
            #     # plt.plot(x, map_soln["pred"], color="C1", lw=3,
            #     #         label="$\mathrm{model}$")
            #     # plt.xlabel("$\mathrm{Time~[days]}$")
            #     # plt.ylabel("$\mathrm{Relative~flux}$")
            #     plt.plot(x, y, "k.", ms=.5, label="data")
            #     plt.plot(x, map_soln["pred"], color="C1", lw=3,
            #             label="model")
            #     plt.xlabel("Time [days]")
            #     plt.ylabel("Relative flux")
            #     plt.legend(fontsize=20)

            # Sample from the posterior
            np.random.seed(42)
            sampler = xo.PyMC3Sampler()
            with model:
                sampler.tune(tune=2000, start=map_soln,
                            step_kwargs=dict(target_accept=0.9), cores=1)
                trace = sampler.sample(draws=2000, cores=1)

            # Save samples
            samples = pm.trace_to_dataframe(trace)
            self.samples = samples

            self.period_samples = trace["period"]
            gp_period = np.median(self.period_samples)
            lower = np.percentile(self.period_samples, 16)
            upper = np.percentile(self.period_samples, 84)
            errm = gp_period - lower
            errp = upper - gp_period
            logQ = np.median(trace["logQ0"])
            upperQ = np.percentile(trace["logQ0"], 84)
            lowerQ = np.percentile(trace["logQ0"], 16)
            Qerrp = upperQ - logQ
            Qerrm = logQ - lowerQ

        self.gp_period = gp_period
        self.errp, self.errm = errp, errm
        self.logQ = logQ
        self.Qerrp, self.Qerrm = Qerrp, Qerrm

        return gp_period, errp, errm, logQ, Qerrp, Qerrm

    def plot_posterior(self):

        # Plot the posterior.
        plt.hist(self.period_samples, 30, histtype="step", color="k")
        plt.axvline(self.gp_period)
        plt.yticks([])
        plt.xlabel("Rotation period [days]")
        plt.ylabel("Posterior density");
        plt.axvline(lower, ls="--", color="C1");
        plt.axvline(upper, ls="--", color="C1");
        plt.xlim(0, 50);
