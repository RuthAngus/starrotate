import numpy as np
import phase_dispersion_minimization as pdm
import matplotlib.pyplot as plt

def test_sj2():
    np.random.seed(42)
    N = 10000
    t = np.linspace(0, .1, N)
    x = np.random.randn(N)
    sj2 = pdm.sj2(x, 0, N)
    assert np.isclose(sj2, 1, atol=.01)

def test_s2():
    np.random.seed(42)
    N = 10000
    M = 10
    nj = np.ones(M) * N
    sj2 = np.zeros(M)
    for j in range(M):
        t = np.linspace(0, .1, N)
        x = np.random.randn(N)
        sj2[j] = pdm.sj2(x, 0, nj[j])
    s2 = pdm.s2(nj, sj2, M)
    assert np.isclose(s2, 1, atol=.01)


def test_phase(t, x):
    phase = pdm.calc_phase(10, t)

    # plt.plot(phase, x, ".")
    # plt.savefig("phase_test")
    # plt.close()


def test_phase_bins(t, x):
    nbins = 100
    phase = pdm.calc_phase(10, t)

    x_means, phase_bins, Ns, sj2s, x_binned, phase_binned = \
        pdm.phase_bins(nbins, phase, x)
    mid_phase_bins = np.diff(phase_bins) * .5 + phase_bins[:-1]

    for j in range(nbins):
        plt.plot(phase_binned[j], x_binned[j], ".", alpha=.1, zorder=0)

    plt.plot(mid_phase_bins, x_means, "ko", zorder=1)
    plt.savefig("phase_test2")
    plt.close()

    assert np.isclose(max(x_means), 1, atol=.02)


def test_phi():

    # Generate some data
    t = np.linspace(0, 100, 1000)
    p = 10
    w = 2*np.pi/p
    x1 = np.sin(w*t)
    x2 = .4*np.sin(w*t + np.pi/2)
    x3 = .3*np.sin(w*t + np.pi/3)
    x = x1 + x2 + x3

    plt.plot(t, x1)
    plt.plot(t, x2)
    plt.plot(t, x3)
    plt.plot(t, x)
    plt.savefig("test")

    # Calculate the Phi statistic over a range of periods
    nperiods = 5000
    periods = np.linspace(1, 21, nperiods)
    phis = np.zeros(nperiods)
    for i, p in enumerate(periods):
        phis[i] = pdm.phi(10, p, x, t)

    # Find period with the lowest Phi
    ind = np.argmin(phis)
    print("best period = ", periods[ind])

    # Calculate the phase at that period (for plotting)
    phase = pdm.calc_phase(periods[ind], t)

    # Make the plot
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(311)
    ax1.plot(t, x, ".")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Flux")

    ax2 = fig.add_subplot(312)
    ax2.plot(phase, x, ".")
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Flux")

    ax3 = fig.add_subplot(313)
    ax3.plot(periods, phis*periods[ind])
    ax3.set_xlabel("Period [days]")
    ax3.set_ylabel("Dispersion")
    ax3.axvline(periods[ind], color="C1")

    fig.savefig("phi_test")

    assert np.isclose(periods[ind], 10, atol=.1)


if __name__ == "__main__":
    test_sj2()
    test_s2()

    # Generate some data
    t = np.linspace(0, 100, 1000)
    p = 10
    w = 2*np.pi/p
    x = np.sin(w*t) + np.random.randn(len(t))*1e-2

    test_phase(t, x)
    test_phase_bins(t, x)
    test_phi()
