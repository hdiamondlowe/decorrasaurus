

from imports import *
import batman
import celerite
from celerite.modeling import Model
from celerite import terms

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
        # get the kernel parameters for each subdirectory
        self.allparaminds = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            n = self.inputs[subdir]['n']

            self.allparaminds.append([])
            for k, klabel in enumerate(self.inputs[subdir]['kernellabels']):
                self.allparaminds[s].append(np.argwhere(np.array(self.wavebin['freeparamnames']) == klabel+str(n))[0][0])

        # set up the model for the transit parameters
        self.calclimbdark = self.limbdarkconversion()
        self.batmandictionaries = []
        self.batmandictionariesfit = []
        self.batmanupdatenames = []
        self.batmanparaminds = []
        self.paramsTransitModel = []
        self.times = []

        firstdir = self.wavebin['subdirectories'][0]
        firstn = self.inputs[firstdir]['n']

        for s, subdir in enumerate(self.wavebin['subdirectories']):

            self.batmandictionaries.append({})
            self.batmandictionariesfit.append({})
            self.batmandictionariesfit[s]['bounds'] = {}
            self.batmanupdatenames.append([])
            self.batmanparaminds.append([])

            n = self.inputs[subdir]['n']
           
            for t, tranlabel in enumerate(self.inputs[subdir]['tranlabels']):

                self.batmandictionaries[s][tranlabel] = self.inputs[subdir]['tranparams'][t]

                if s == 0:
                    if tranlabel+firstn in self.wavebin['freeparamnames']: self.paramsTransitModel.append(tranlabel)

                if tranlabel+n in self.wavebin['freeparamnames']:
                    paramind = np.argwhere(self.wavebin['freeparamnames'] == tranlabel+str(n))[0][0]
                elif (tranlabel in self.inputs['jointparams']) and (tranlabel+firstn in self.wavebin['freeparamnames']):
                    paramind =  np.argwhere(np.array(self.wavebin['freeparamnames']) == tranlabel+firstn)[0][0]
                else: continue

                self.batmanupdatenames[s].append(tranlabel)
                self.batmanparaminds[s].append(paramind)
                self.allparaminds[s].append(paramind)
                self.batmandictionariesfit[s][tranlabel] = self.inputs[subdir]['tranparams'][t]
                if self.inputs[subdir]['tranbounds'][t][0] == True: bound0 = None
                else: bound0 = self.inputs[subdir]['tranbounds'][t][0]
                if self.inputs[subdir]['tranbounds'][t][1] == True: bound1 = None
                else: bound1 = self.inputs[subdir]['tranbounds'][t][1]
                self.batmandictionariesfit[s]['bounds'][tranlabel] = (bound0, bound1)

            # make times such that time of mid-transit should be at 0
            self.times.append((self.wavebin[subdir]['compcube']['bjd'] - self.inputs[subdir]['toff'])[self.wavebin[subdir]['binnedok']])

        self.batmanparams = []
        self.batmanmodels = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):

            setupbatmanparams, setupbatmanmodel = self.setup_batman_model(self.batmandictionaries[s], self.times[s])
            self.batmanparams.append(setupbatmanparams)
            self.batmanmodels.append(setupbatmanmodel)


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

    def makemodelGP(self, params):

        '''
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
                
                try: self.batmanparams.u = self.calclimbdark(self.u0, self.u1)
                except(AttributeError): self.batmanparams.u = self.calclimbdark(self.batmandictionary['u0'], self.batmandictionary['u1'])   #limb darkening coefficients


                # now return a light curve
                return self.batmanmodel.light_curve(self.batmanparams)

        mean_models = [meanTransitModel(self.batmandictionaries[i], self.batmanmodels[i], self.batmanparams[i], self.calclimbdark, self.batmandictionariesfit[i]) for i in self.rangeofdirectories]
        '''
        batmanparams = batman.TransitParams()
        batmanparams.t0 = 0
        batmanparams.per = 24.73712
        batmanparams.rp = 0.074
        batmanparams.a = 95.34
        batmanparams.inc = 89.89
        batmanparams.ecc = 0
        batmanparams.w = 90
        batmanparams.limb_dark = 'quadratic'
        batmanparams.u = [.5, .5] #[0.30947, 0.31279]

        batmanmodel = batman.TransitModel(batmanparams, self.times[0])


        class TransitModel(Model):

            parameter_names = ('dt', 'rp', 'u0', 'u1')#, 'airmasscoeff', 'offset')

            def get_value(self, t):

                #lc = compf(t)/targf(t)

                batmanparams.t0 = self.dt
                batmanparams.rp = self.rp
                batmanparams.u = [self.u0, self.u1]

               #quadratic = self.m*t**2

                #batmanmodel = batman.TransitModel(batmanparams, t)

                # airmass model
                #airmassterm = self.airmasscoeff*parabolatrend + self.offset

                return batmanmodel.light_curve(batmanparams) #* quadratic

        freeparams = dict(dt=batmanparams.t0, rp=batmanparams.rp, u0=batmanparams.u[0], u1=batmanparams.u[1])#, m=-1.5)#, airmasscoeff=1.0, offset=1.0)
        freeparams['bounds'] = dict(dt=(-0.0010376, 0.003059236), rp=(0.0001, 0.1), u0=(0.1, 0.999), u1=(0.1, 0.999))#, airmasscoeff=(-10, 10), offset=(-10, 10))
        #freeparams['bounds'] = dict(dt=(-0.0010376, 0.003059236), rp=(0.02, 0.08), u0=(0.1, 0.999), u1=(0.1, 0.999))#, airmasscoeff=(-10, 10), offset=(-10, 10))

        mean_model = TransitModel(**freeparams)

        gps = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            kernelparams = self.inputs[subdir]['kernelparams']
            kernelbounds = np.array(self.inputs[subdir]['kernelbounds']).T
            print(kernelparams, kernelbounds)
            if self.inputs[subdir]['kernelname'] == 'RealTerm':
                kernel = terms.RealTerm(log_a=kernelparams[0], log_c=kernelparams[1], 
                               bounds=dict(log_a=(kernelbounds[0][0], kernelbounds[0][1]), log_c=(kernelbounds[1][0], kernelbounds[1][1])))
            elif self.inputs[subdir]['kernelname'] == 'ComplexTerm':
                kernel = terms.ComplexTerm(log_a=kernelparams[0], log_b=kernelparams[1], log_c=kernelparams[2], log_d=kernelparams[3], 
                               bounds=dict(log_a=(kernelbounds[0][0], kernelbounds[0][1]), log_b=(kernelbounds[1][0], kernelbounds[1][1]),
                                           log_c=(kernelbounds[2][0], kernelbounds[2][1]), log_d=(kernelbounds[3][0], kernelbounds[3][1])))
            elif self.inputs[subdir]['kernelname'] == 'SHOTerm':
                kernel = terms.SHOTerm(log_S0=kernelparams[0], log_Q=kernelparams[1], log_omega0=kernelparams[2], 
                               bounds=dict(log_S0=(kernelbounds[0][0], kernelbounds[0][1]), log_Q=(kernelbounds[1][0], kernelbounds[1][1]),
                                           log_omega0=(kernelbounds[2][0], kernelbounds[2][1])))
            elif self.inputs[subdir]['kernelname'] == 'Matern32Term':
                print('using matern32term')
                kernel = terms.Matern32Term(log_sigma=kernelparams[0], log_rho=kernelparams[1], 
                               bounds=dict(log_sigma=(kernelbounds[0][0], kernelbounds[0][1]), log_rho=(kernelbounds[1][0], kernelbounds[1][1])))
            print('kernel', kernel)

            gp = celerite.GP(kernel=kernel, mean=mean_model, fit_mean=True)
            gp.compute(self.times[s])
            gps.append(gp)

        return gps
        


