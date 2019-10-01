

from imports import *
import batman

class ModelMaker(Talker):

    def __init__(self, inputs, wavebin):
        ''' initialize the model maker'''
        Talker.__init__(self)

        self.inputs = inputs
        self.wavebin = wavebin
        self.rangeofdirectories = range(len(self.wavebin['subdirectories']))

        # set up the model for the systematic parameters
        # determine the time arrays, systematic parameter arrays, and what indices go into polyparaminds or sysparaminds
        self.timearrays = []
        self.sysparamarrays = []
        self.polyparaminds = []
        self.sysparaminds = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            n = self.inputs[subdir]['n']

            time_range = self.wavebin[subdir]['compcube']['bjd'][self.wavebin[subdir]['binnedok']][-1] - self.wavebin[subdir]['compcube']['bjd'][self.wavebin[subdir]['binnedok']][0]
            self.timearrays.append((self.wavebin[subdir]['compcube']['bjd']-self.inputs[subdir]['toff'])/time_range)
            
            self.polyparaminds.append([])
            for p, plabel in enumerate(self.inputs[subdir]['polylabels']):
                self.polyparaminds[s].append(np.argwhere(np.array(self.wavebin['freeparamnames']) == plabel+str(n))[0][0])

            self.sysparaminds.append([])
            self.sysparamarrays.append([])
            for flabel in self.inputs[subdir]['fitlabels']:
                self.sysparaminds[s].append(np.argwhere(np.array(self.wavebin['freeparamnames']) == flabel+str(n))[0][0])
                self.sysparamarrays[s].append(np.array(self.wavebin[subdir]['compcube'][flabel])) 
            self.sysparamarrays[s] = np.array(self.sysparamarrays[s])

        self.ones = np.ones(len(self.wavebin['subdirectories']))

        # set up the model for the transit parameters
        self.calclimbdark = self.limbdarkconversion()
        self.batmandictionaries = []
        self.batmanupdatenames = []
        self.batmanparaminds = []
        self.batmanparams = []
        self.batmanmodels = []

        firstdir = self.wavebin['subdirectories'][0]
        firstn = self.inputs[firstdir]['n']

        for s, subdir in enumerate(self.wavebin['subdirectories']):

            self.batmandictionaries.append({})
            self.batmanupdatenames.append([])
            self.batmanparaminds.append([])

            n = self.inputs[subdir]['n']
           
            for t, tranlabel in enumerate(self.inputs[subdir]['tranlabels']):

                self.batmandictionaries[s][tranlabel] = self.inputs[subdir]['tranparams'][t]

                if tranlabel+n in self.wavebin['freeparamnames']:
                    paramind = np.argwhere(self.wavebin['freeparamnames'] == tranlabel+str(n))[0][0]
                    self.batmanupdatenames[s].append(tranlabel)
                    self.batmanparaminds[s].append(paramind)
                elif (tranlabel in self.inputs['jointparams']) and (tranlabel+firstn in self.wavebin['freeparamnames']):
                    paramind =  np.argwhere(np.array(self.wavebin['freeparamnames']) == tranlabel+firstn)[0][0]
                    self.batmanupdatenames[s].append(tranlabel)
                    self.batmanparaminds[s].append(paramind)

            # make times such that time of mid-transit should be at 0
            times = self.wavebin[subdir]['compcube']['bjd'] - self.inputs[subdir]['toff']
            setupbatmanparams, setupbatmanmodel = self.setup_batman_model(self.batmandictionaries[s], times)
            self.batmanparams.append(setupbatmanparams)
            self.batmanmodels.append(setupbatmanmodel)


    def limbdarkconversion(self):

        if self.inputs['ldlaw'] == 'sq':
            def calc_u0u1(v0, v1):
                u0 = (75./34.)*v0 + (45./34.)*v1
                u1 = (45./34.)*v0 - (75./34.)*v1
                return [u0, u1]

        elif self.inputs['ldlaw'] == 'qd':
            def calc_u0u1(v0, v1):
                u0 = (2./5.)*v0 + (1./5.)*v1
                u1 = (1./5.)*v0 - (2./5.)*v1
                return [u0, u1]

        return calc_u0u1

    def setup_batman_model(self, dictionary, times):

        batmanparams = batman.TransitParams()       #object to store transit parameters
        batmanparams.t0 = dictionary['dt']           #time of inferior conjunction
        batmanparams.per = dictionary['per']                 #orbital period [days] (from Jason's Spitzer data: 1.62895579)
        batmanparams.rp = dictionary['rp']                   #planet radius (in units of stellar radii)
        batmanparams.a = dictionary['a']                     #semi-major axis (in units of stellar radii)
        batmanparams.inc = dictionary['inc']                 #orbital inclination (in degrees)']
        batmanparams.ecc = dictionary['ecc']                 #eccentricity
        batmanparams.w = dictionary['omega']                        #longitude of periastron (in degrees)
        if self.inputs['ldlaw'] == 'qd': batmanparams.limb_dark = "quadratic"        #limb darkening model
        elif self.inputs['ldlaw'] == 'sq': batmanparams.limb_dark = "squareroot"        #limb darkening model
        
        batmanparams.u = self.calclimbdark(dictionary['u0'], dictionary['u1'])                   #limb darkening coefficients
        #params.fac = 1e-3

        batmanmodel = batman.TransitModel(batmanparams, times)    #initializes model
        #model = batman.TransitModel(params, self.times, fac=self.batmanfac)    #initializes model

        # return the model and parameter structures so we can update them later to get the model light curves we want
        return batmanparams, batmanmodel

    def update_batman_model(self, batmanparams, batmanmodel, dictionary, updates, params):

        [dictionary.update({updates[i]: params[i]}) for i in range(len(params))]

        batmanparams.t0 = dictionary['dt']           #time of inferior conjunction
        batmanparams.per = dictionary['per']                 #orbital period [days] (from Jason's Spitzer data: 1.62895579)
        batmanparams.rp = dictionary['rp']                   #planet radius (in units of stellar radii)
        batmanparams.a = dictionary['a']                     #semi-major axis (in units of stellar radii)
        batmanparams.inc = dictionary['inc']                 #orbital inclination (in degrees)
        batmanparams.ecc = dictionary['ecc']                 #eccentricity
        batmanparams.w = dictionary['omega']                 #longitude of periastron (in degrees)
        
        batmanparams.u = self.calclimbdark(dictionary['u0'], dictionary['u1'])                    #limb darkening coefficients
        #params.fac = 1e-3

        # now return a light curve
        return np.array(batmanmodel.light_curve(batmanparams))

    
    def makemodel(self, params):

        # make the model that fits to the data systematics
        polyfuncs = [np.poly1d(params[inds][::-1]) for inds in self.polyparaminds]
        polymodels = [polyfuncs[i](self.timearrays[i]) for i in self.rangeofdirectories]
        sysmodels = [np.sum(np.multiply(params[self.sysparaminds[i]], self.sysparamarrays[i].T).T, axis=0) for i in self.rangeofdirectories]
        self.fitmodel = np.sum([polymodels, sysmodels, self.ones], axis=0)

        # make the transit models wiht batman
        self.batmanmodel = [self.update_batman_model(self.batmanparams[i], self.batmanmodels[i], self.batmandictionaries[i], self.batmanupdatenames[i], params[self.batmanparaminds[i]]) for i in self.rangeofdirectories]

        return [self.fitmodel[i]*self.batmanmodel[i] for i in self.rangeofdirectories]


