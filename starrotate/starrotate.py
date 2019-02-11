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

    def __init__(self, raw_time, raw_flux, raw_flux_err, starname, plot=True,
                 plot_path="."):

        self.raw_time = raw_time
        self.raw_flux = raw_flux
        self.raw_flux_err = raw_flux_err
        self.starname = starname
        self.plot = plot
        self.plot_path = plot_path

    def process_light_curve(self, t0=None, dur=None, porb=None, zoom=100):

        self.t0, self.dur, self.porb = t0, dur, porb

        # Plot the light curve
        plt.figure(figsize=(20, 5))
        plt.plot(self.raw_time, self.raw_flux, "k.", ms=.5);
        plt.xlabel("Time [days]")
        plt.ylabel("Relative Flux");
        plt.subplots_adjust(bottom=.2)
        plt.savefig("{0}/{1}_full_lightcurve".format(self.plot_path,
                                                     self.starname))
        plt.close()

        # Mask the transits and plot masked light curve
        if t0 is not None and dur is not None and porb is not None:
            transit = self.transit_mask()
            time, flux, flux_err = self.raw_time[~transit], \
                self.raw_flux[~transit], self.raw_flux_err[~transit]

            if self.plot:
                ms = 5
                plt.figure(figsize=(20, 5))
                plt.plot(self.raw_time, self.raw_flux, ".", color="C1", ms=ms,
                         zorder=0);
                plt.plot(time, flux, ".", color="C0", ms=ms, zorder=1);
                plt.xlabel("Time [days]")
                plt.ylabel("Relative Flux");
                plt.subplots_adjust(bottom=.2)
                plt.savefig("{0}/{1}_masked_lightcurve".format(self.plot_path,
                                                               self.starname))
                plt.close()

        self.time, self.flux, self.flux_err = time, flux, flux_err

        # Plot a zoom in.
        if self.plot:
            plt.figure(figsize=(20, 5))
            plt.plot(self.raw_time, self.raw_flux, ".", color="C1", ms=2,
                     zorder=0);
            plt.plot(self.time, self.flux, ".", color="C0", ms=2, zorder=1);
            # plt.xlabel("$\mathrm{Time~[days]}$")
            # plt.ylabel("$\mathrm{Relative~Flux}$");
            plt.xlabel("Time [days]")
            plt.ylabel("Relative Flux");
            plt.xlim(self.time[0], self.time[0] + zoom)
            plt.subplots_adjust(bottom=.15)
            plt.savefig("{0}/{1}_zoom".format(self.plot_path, self.starname))
            plt.close()

        # Save the processed light curve data.
        lc = pd.DataFrame(dict({"time": self.time, "flux": self.flux,
                                "flux_err": self.flux_err}))
        lc.to_csv("{0}/{1}_lc_data.csv".format(self.plot_path, self.starname))

#         # Calculate the rotation period.
#         ls_period = self.LS_rotation()
#         acf_period = self.ACF_rotation()
#         gp_period = self.GP_rotation()
#         return gp_period, ls_period, acf_period

    def transit_mask(self):
        # # How many transits?
        # ntransit = int((self.raw_time[-1] - self.raw_time[0])//self.porb)
        # transit = (self.t0 - .5*self.dur < self.raw_time) * \
        #     (self.raw_time < self.t0 + .5*self.dur)
        # for i in range(ntransit):
        #     transit += (self.t0 + i*self.porb - .5*self.dur < self.raw_time)\
        #         * (self.raw_time < self.t0 + i*self.porb + .5*self.dur)

        t0 = float(self.t0) % self.porb
        dur = float(self.dur) / 24.

        m = np.abs((self.raw_time - t0 + 0.5*self.porb) \
                   % self.porb - 0.5*self.porb) < 1.5*dur
        return m

    def fold_plot(self, period, method):
        x_fold = ((self.time - self.time[0]) % period)/period
        plt.figure(figsize=(12, 9))
        plt.plot(x_fold, self.flux, ".k", ms=1);
        plt.ylim(-np.std(self.flux)*3, np.std(self.flux)*3)
        # plt.xlabel("$\mathrm{Phase}$")
        # plt.ylabel("$\mathrm{Normalized~Flux}$")
        plt.xlabel("Phase")
        plt.ylabel("Normalized Flux")

        bins = np.linspace(0, 1, 20)
        denom, _ = np.histogram(x_fold, bins)
        num, _ = np.histogram(x_fold, bins, weights=self.flux)
        denom[num == 0] = 1.0
        plt.plot(0.5*(bins[1:] + bins[:-1]), num / denom, ".-", color="C1",
                ms=10);
        plt.subplots_adjust(left=.15, bottom=.15)
        plt.savefig("{0}/{1}_{2}_fold".format(self.plot_path, self.starname,
                                              method))
        plt.close()

    def LS_rotation(self):

        # Calculate a LS period
        results = xo.estimators.lomb_scargle_estimator(
            self.time, self.flux, max_peaks=1, min_period=1.0,
            max_period=10.0, samples_per_peak=50)

        # Make a plot of the periodogram.
        if self.plot:
            plt.figure(figsize=(16, 9))
            peak = results["peaks"][0]
            freq, power = results["periodogram"]
            plt.plot(-np.log10(freq), power, "k", zorder=0)
            plt.axvline(np.log10(peak["period"]), color="C1", lw=4, alpha=0.5,
                        zorder=1)
            plt.xlim((-np.log10(freq)).min(), (-np.log10(freq)).max())
            plt.yticks([])
            # plt.xlabel("$\log_{10}(\mathrm{Period [days]}$)")
            # plt.ylabel("$\mathrm{Power}$");
            plt.xlabel("log10(Period [days])")
            plt.ylabel("Power");
            plt.subplots_adjust(left=.15, bottom=.15)
            plt.savefig("{0}/{1}_pgram".format(self.plot_path, self.starname))
            plt.close()

            # Plot the light curve folded on this period.
            self.fold_plot(peak["period"], "LS")

        # Save the periodogram data.
        pgram = pd.DataFrame(dict({"freq": freq, "power": power, "period":
                                   peak["period"]}))
        pgram.to_csv("{0}/{1}_pgram_data.csv".format(self.plot_path,
                                                     self.starname))

        self.ls_period = peak["period"]
        return peak["period"]

    def ACF_rotation(self):
        lags, acf, acf_period = simple_acf(self.time, self.flux)

        if self.plot:
            plt.figure(figsize=(16, 9))
            plt.plot(lags, acf, "k")
            plt.axvline(acf_period, color="C1")
            # plt.xlabel("$\mathrm{Period~[days]}$")
            # plt.ylabel("$\mathrm{Correlation}$")
            plt.xlabel("Period [days]")
            plt.ylabel("Correlation")
            plt.xlim(0, 20)
            plt.subplots_adjust(left=.15, bottom=.15)
            plt.savefig("{0}/{1}_acf".format(self.plot_path, self.starname))
            plt.close()

            # Plot the folded light curve
            self.fold_plot(acf_period, method="ACF")

        # Save the ACF data.
        ACF = pd.DataFrame(dict({"lags": lags, "acf": acf, "period":
                                 acf_period}))
        ACF.to_csv("{0}/{1}_acf_data.csv".format(self.plot_path,
                                                 self.starname))

        self.acf_period = acf_period
        return acf_period

    def GP_rotation(self):
        x = self.time
        y = self.flux
        yerr = self.flux_err

        with pm.Model() as model:

            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sd=10.0)

            # A jitter term describing excess white noise
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(yerr)), sd=5.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
            logperiod = pm.Normal("logperiod", mu=np.log(self.ls_period),
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
            map_soln = pm.find_MAP(start=model.test_point)

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
            samples.to_hdf("{0}/{1}_samples.h5".format(self.plot_path,
                                                       self.starname),
                           "trace")

            period_samples = trace["period"]
            gp_period = np.median(period_samples)
            lower = np.percentile(period_samples, 16)
            upper = np.percentile(period_samples, 84)
            errm = gp_period - lower
            errp = upper - gp_period
            logQ = np.median(trace["logQ0"])
            upperQ = np.percentile(trace["logQ0"], 84)
            lowerQ = np.percentile(trace["logQ0"], 16)
            Qerrp = upperQ - logQ
            Qerrm = logQ - lowerQ

            # Plot the posterior.
            if self.plot:
                plt.hist(period_samples, 30, histtype="step", color="k")
                plt.axvline(gp_period)
                plt.yticks([])
                # plt.xlabel("$\mathrm{Rotation~period~[days]}$")
                # plt.ylabel("$\mathrm{Posterior~density}$");
                plt.xlabel("Rotation period [days]")
                plt.ylabel("Posterior density");
                plt.axvline(lower, ls="--", color="C1");
                plt.axvline(upper, ls="--", color="C1");
                plt.xlim(0, 50);

                # Plot the folded light curve
                self.fold_plot(gp_period, method="GP")

        self.gp_period = gp_period
        self.errp, self.errm = errp, errm
        self.logQ = logQ
        self.Qerrp, self.Qerrm = Qerrp, Qerrm

        return gp_period, errp, errm, logQ, Qerrp, Qerrm
