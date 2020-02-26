

from imports import *
import batman
import george
from george.modeling import Model
from scipy.optimize import minimize
import dill as pickle

class ModelMaker(Talker):

    def __init__(self, inputs, wavebin):
        ''' initialize the model maker'''
        Talker.__init__(self)

        self.inputs = inputs.inputs
        self.wavebin = wavebin
        self.rangeofdirectories = range(len(self.wavebin['subdirectories']))
        if self.inputs['sysmodel'] == 'linear': self.setupLinear(inputs)
        elif self.inputs['sysmodel'] == 'GP': self.setupGP()

    def setupLinear(self, inputs):

        # set up the model for the systematic parameters
        # determine the time arrays, systematic parameter arrays, and what indices go into polyparaminds or sysparaminds
        timearrays = []
        sysparamlists = []
        polyparaminds = []
        sysparaminds = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            n = self.inputs[subdir]['n']

            time_range = self.wavebin[subdir]['compcube']['bjd'][self.wavebin[subdir]['binnedok']][-1] - self.wavebin[subdir]['compcube']['bjd'][self.wavebin[subdir]['binnedok']][0]
            timearrays.append((self.wavebin[subdir]['compcube']['bjd']-self.inputs[subdir]['toff'])/time_range)
            
            polyparaminds.append([])
            for p, plabel in enumerate(self.inputs[subdir]['polylabels']):
                polyparaminds[s].append(np.argwhere(np.array(self.wavebin['freeparamnames']) == plabel+str(n))[0][0])

            sysparaminds.append([])
            sysparamlists.append([])
            for flabel in self.inputs[subdir]['fitlabels']:
                sysparaminds[s].append(np.argwhere(np.array(self.wavebin['freeparamnames']) == flabel+str(n))[0][0])
                sysparamlists[s].append(np.array(self.wavebin[subdir]['compcube'][flabel])) 

        # transform everything into a numpy array so we can do numpy math
        self.timearrays = inputs.equalizeArrays1D(timearrays)
        self.sysparamarrays = inputs.equalizeArrays2D(sysparamlists).T
        self.polyparaminds = polyparaminds              # keep as uneven list to ensure that polynomials are correct length
        self.sysparaminds = inputs.equalizeArrays1D(sysparaminds).astype(int).T

        self.ones = np.ones(len(self.wavebin['subdirectories']))

        # set up the model for the transit parameters
        self.calclimbdark = self.limbdarkconversion()
        self.batmandictionaries = []
        self.batmanupdatenames = []
        self.batmanparaminds = []
        times = []

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
                elif (tranlabel in self.inputs['jointparams']) and (tranlabel+firstn in self.wavebin['freeparamnames']):
                    paramind =  np.argwhere(np.array(self.wavebin['freeparamnames']) == tranlabel+firstn)[0][0]
                else: continue

                self.batmanupdatenames[s].append(tranlabel)
                self.batmanparaminds[s].append(paramind)

            # make times such that time of mid-transit should be at 0
            times.append(self.wavebin[subdir]['compcube']['bjd'] - self.inputs[subdir]['toff'])

        times = inputs.equalizeArrays1D(times, padwith=1)
        self.batmanparams = []
        self.batmanmodels = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):

            setupbatmanparams, setupbatmanmodel = self.setup_batman_model(self.batmandictionaries[s], times[s])
            self.batmanparams.append(setupbatmanparams)
            self.batmanmodels.append(setupbatmanmodel)

    def setupGP(self):

        self.allparaminds = []

        # set up the model for the transit parameters
        self.calclimbdark = self.limbdarkconversion()
        self.batmandictionaries = []
        self.batmandictionariesfit = []
        self.batmanparaminds = []
        self.paramsTransitModel = []
        self.times = []
        self.kernels = []
        self.whitenoise = []

        firstdir = self.wavebin['subdirectories'][0]
        firstn = self.inputs[firstdir]['n']

        for s, subdir in enumerate(self.wavebin['subdirectories']):

            self.allparaminds.append([])

            self.batmandictionaries.append({})
            self.batmandictionariesfit.append({})
            self.batmandictionariesfit[s]['bounds'] = {}
            self.batmanparaminds.append([])

            n = self.inputs[subdir]['n']

            self.kernels.append(self.wavebin[subdir]['gpkernel'])
            self.whitenoise.append(self.wavebin[subdir]['gpwhitenoise'])
            
            #print(self.inputs[subdir]['tranbounds'])           

            for t, tranlabel in enumerate(self.inputs[subdir]['tranlabels']):

                self.batmandictionaries[s][tranlabel] = self.inputs[subdir]['tranparams'][t]

                if s == 0:
                    if tranlabel+firstn in self.wavebin['freeparamnames']: self.paramsTransitModel.append(tranlabel)

                if tranlabel+n in self.wavebin['freeparamnames']:
                    paramind = np.argwhere(self.wavebin['freeparamnames'] == tranlabel+n)[0][0]
                elif (tranlabel in self.inputs['jointparams']) and (tranlabel+firstn in self.wavebin['freeparamnames']):
                    paramind =  np.argwhere(np.array(self.wavebin['freeparamnames']) == tranlabel+firstn)[0][0]
                else: continue

                # past here in the loop, everything should be a free parameter!
                self.batmanparaminds[s].append(paramind)
                self.allparaminds[s].append(paramind)
                self.batmandictionariesfit[s][tranlabel] = self.wavebin['freeparamvalues'][paramind]

                boundlo = self.wavebin['freeparambounds'][0][paramind]
                boundhi = self.wavebin['freeparambounds'][1][paramind]
                self.batmandictionariesfit[s]['bounds'][tranlabel] = (boundlo, boundhi)

            whitenoiseparamind = np.argwhere(self.wavebin['freeparamnames'] == 'whitenoise'+n)[0][0]
            self.allparaminds[s].append(whitenoiseparamind)

            for k, klabel in enumerate(self.wavebin[subdir]['kernellabels']):
                self.allparaminds[s].append(np.argwhere(np.array(self.wavebin['freeparamnames']) == klabel+n)[0][0])

            # make times such that time of mid-transit should be at 0
            self.times.append((self.wavebin[subdir]['compcube']['bjd'] - self.inputs[subdir]['toff'])[self.wavebin[subdir]['binnedok']])


        #print(self.batmandictionariesfit)

        self.batmanparams = []
        self.batmanmodels = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):

            setupbatmanparams, setupbatmanmodel = self.setup_batman_model(self.batmandictionaries[s], self.times[s])
            self.batmanparams.append(setupbatmanparams)
            self.batmanmodels.append(setupbatmanmodel)

        self.lcs = [self.wavebin[subdir]['lc'] for subdir in self.wavebin['subdirectories']]
        self.photnoiseest = [self.wavebin[subdir]['photnoiseest'] for subdir in self.wavebin['subdirectories']]
        self.binnedok = [self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']]

    def limbdarkconversion(self):

        # convert back to u0 and u1 to make the light curve; re-parameterization from Kipping+ (2013)

        if self.inputs['ldlaw'] == 'sq':
            def calc_u0u1(q0, q1):
                u0 = np.sqrt(q0)*(1 - 2*q1)
                u1 = 2*np.sqrt(q0)*q1
                return [u0, u1]

        elif self.inputs['ldlaw'] == 'qd':
            def calc_u0u1(q0, q1):
                u0 = 2*np.sqrt(q0)*q1
                u1 = np.sqrt(q0)*(1 - 2*q1)
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

        batmanmodel = batman.TransitModel(batmanparams, times)    #initializes model

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
        
        batmanparams.u = self.calclimbdark(dictionary['u0'], dictionary['u1'])   #limb darkening coefficients

        # now return a light curve
        return np.array(batmanmodel.light_curve(batmanparams))

   
    def makemodelLinear(self, params):

        # make the model that fits to the data systematics
        # this involves making a 1d polynomial for each data set and applying it to the time array for each data set
        Lpolymodels = [np.polynomial.legendre.Legendre(params[self.polyparaminds[i]])(self.timearrays[i]) for i in self.rangeofdirectories]

        sysmodels = np.sum(np.multiply(params[self.sysparaminds], self.sysparamarrays), axis=1)
        self.fitmodel = np.sum([np.array(Lpolymodels).T, sysmodels, self.ones]).T

        # make the transit models wiht batman
        self.batmanmodel = [self.update_batman_model(self.batmanparams[i], self.batmanmodels[i], self.batmandictionaries[i], self.batmanupdatenames[i], params[self.batmanparaminds[i]]) for i in self.rangeofdirectories]

        return np.multiply(self.fitmodel, self.batmanmodel)

    def makemodelGP(self):


        class meanTransitModel(Model):
            parameter_names = tuple(self.paramsTransitModel)#, 'u0', 'u1')#, 'airmasscoeff', 'offset')

            def __init__(self, batmandictionary, batmanmodel, batmanparams, calclimbdark, batmandictionariesfit):
                # inherit the __init__ class from the parent class Molde
                Model.__init__(self, **batmandictionariesfit)

                #print(parameter_names)
                self.batmandictionary = batmandictionary
                self.batmanmodel = batmanmodel
                self.batmanparams = batmanparams
                self.calclimbdark = calclimbdark

            def get_value(self, t):#, batmanparams, batmanmodel):

                try: self.batmanparams.t0 = self.dt           #time of inferior conjunction
                except(AttributeError): self.batmanparams.t0 = self.batmandictionary['dt']
                try: self.batmanparams.per = self.per                #orbital period [days] (from Jason's Spitzer data: 1.62895579)
                except(AttributeError): self.batmanparams.per = self.batmandictionary['per']
                try: self.batmanparams.rp = self.rp
                except(AttributeError): self.batmanparams.rp = self.batmandictionary['rp']                   #planet radius (in units of stellar radii)
                try: self.batmanparams.a = self.a
                except(AttributeError): self.batmanparams.a = self.batmandictionary['a']                     #semi-major axis (in units of stellar radii)
                try: self.batmanparams.inc = self.inc
                except(AttributeError): self.batmanparams.inc = self.batmandictionary['inc']                 #orbital inclination (in degrees)
                try: self.batmanparams.ecc = self.ecc
                except(AttributeError): self.batmanparams.ecc = self.batmandictionary['ecc']                 #eccentricity
                try: self.batmanparams.w = self.w
                except(AttributeError): self.batmanparams.w = self.batmandictionary['omega']                 #longitude of periastron (in degrees)
                
                try: 
                    self.batmanparams.u = self.calclimbdark(self.u0, self.u1)
                    #print('u0, u1:', self.calclimbdark(self.u0, self.u1))
                except(AttributeError): 
                    self.batmanparams.u = self.calclimbdark(self.batmandictionary['u0'], self.batmandictionary['u1'])   #limb darkening coefficients


                # now return a light curve
                return self.batmanmodel.light_curve(self.batmanparams)

        mean_models = [meanTransitModel(self.batmandictionaries[i], self.batmanmodels[i], self.batmanparams[i], self.calclimbdark, self.batmandictionariesfit[i]) for i in self.rangeofdirectories]
        

        gps = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):

            #gp = george.GP(kernel=self.kernels[s], mean=mean_models[s], fit_mean=True)
            gp = george.GP(kernel=self.kernels[s], mean=mean_models[s], fit_mean=True, white_noise=self.whitenoise[s], fit_white_noise=True)
            gp.compute(self.wavebin[subdir]['gpregressor_arrays'].T[self.wavebin[subdir]['binnedok']], self.photnoiseest[s][self.wavebin[subdir]['binnedok']])

            print(gp.get_parameter_dict())
            print(gp.get_parameter_bounds())

            gps.append(gp)


        return gps

