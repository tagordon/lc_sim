import numpy as np
import celerite
from celerite import terms
import batman

def sim_lc(t, ppm, per_aper_ratio, periodic_freq, aperiodic_freq, A, planet=False, mean=1.0, planet_params=[0, 0, 0, 0, 0, 0, 0, 0, 0]):

    if not planet:
        mean=mean
    if planet:
        mean=transit_model(*planet_params, t)

    aperiodic_sigma = 1/per_aper_ratio
    periodic_sigma = 1
    white_noise_sigma = ppm/1e6

    # set up a gaussian process with two components and white noise

    # non-periodic component
    Q = 1. / np.sqrt(2.0)  # related to the frequency of the variability
    w0 = 2*np.pi*aperiodic_freq
    S0 = (aperiodic_sigma**2.) / (w0 * Q)
    bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0), bounds=bounds)

    # periodic component
    Q = 1.0
    w0 = 2*np.pi*periodic_freq
    S0 = (periodic_sigma**2.) / (w0 * Q)
    kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0), bounds=bounds)

    # white noise
    # kernel += terms.JitterTerm(log_sigma=np.log(white_noise_sigma), bounds=dict(log_sigma=(-15,15)))

    gp = celerite.GP(kernel, mean=0, fit_mean=True, fit_white_noise=True)
    gp.compute(t, white_noise_sigma)

    return A * gp.sample() + mean + white_noise_sigma * np.random.randn(len(t))

def transit_model(t0, per, ppm, a, inc, ecc, w, u0, u1, t):
    rp = np.sqrt(ppm/1e6)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, u0, u1 = t0, per, rp, a, inc, ecc, w, u0, u1
    p.u = [u0, u1]
    p.limb_dark = "quadratic"
    planet = batman.TransitModel(p, t)
    return planet.light_curve(p)
