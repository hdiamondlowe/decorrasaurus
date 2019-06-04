# Make a robin lightcurve
# Author: Hannah Diamond-Lowe
# Date: 11 Feb 2019

import robin
import numpy as np
import astrotools.orbitparams as OP

class RobinLC(object):

    def __init__(self, times, t0, p0, p1, per, inc, a, ecc, omega, u0, u1, ldlaw):
        self.times = times
        self.t0    = t0
        self.p0    = p0
        self.p1    = p1
        self.per   = per
        self.inc   = inc
        self.a     = a
        self.ecc   = ecc
        self.omega = omega
        self.u     = [u0, u1]
        self.ldlaw = ldlaw

    def robin_model(self):

        params = robin.TransitParams()       #object to store transit parameters
        params.t0 = self.t0                   #time of inferior conjunction
        params.per = self.per                 #orbital period [days] (from Jason's Spitzer data: 1.62895579)
        params.p0 = self.p0
        params.p1 = self.p1
        params.a = self.a                     #semi-major axis (in units of stellar radii)
        params.inc = self.inc                 #orbital inclination (in degrees)
        params.ecc = self.ecc                 #eccentricity
        params.w = self.omega                        #longitude of periastron (in degrees)
        if self.ldlaw == 'qd': params.limb_dark = "quadratic"        #limb darkening model
        elif self.ldlaw == 'sq': params.limb_dark = "squareroot"        #limb darkening model
        
        params.u = self.u                     #limb darkening coefficients
        #params.fac = 1e-3

        model = robin.TransitModel(params, self.times)    #initializes model
        #model = batman.TransitModel(params, self.times, fac=self.batmanfac)    #initializes model

        return model.light_curve(params)
