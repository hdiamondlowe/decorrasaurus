# Make a batman lightcurve
# Author: Hannah Diamond-Lowe
# Date: 30 May 2016

import batman
import numpy as np
import astrotools.orbitparams as OP

class BatmanLC(object):

    def __init__(self, times, t0, rp, per, inc, a, ecc, u0, u1):
        self.times = times
        self.t0    = t0
        self.rp    = rp
        self.per   = per
        self.inc   = inc
        self.a     = a
        self.ecc   = ecc
        self.u     = [u0, u1]

    def batman_model(self):

        params = batman.TransitParams()       #object to store transit parameters
        params.t0 = self.t0                   #time of inferior conjunction
        params.per = self.per                 #orbital period [days] (from Jason's Spitzer data: 1.62895579)
        params.rp = self.rp                   #planet radius (in units of stellar radii)
        params.a = self.a                     #semi-major axis (in units of stellar radii)
        params.inc = self.inc                 #orbital inclination (in degrees)
        params.ecc = self.ecc                 #eccentricity
        params.w = 90.                        #longitude of periastron (in degrees)
        params.limb_dark = "quadratic"        #limb darkening model
        params.u = self.u                     #limb darkening coefficients
        #params.fac = 1e-3

        model = batman.TransitModel(params, self.times)    #initializes model

        return model.light_curve(params)
