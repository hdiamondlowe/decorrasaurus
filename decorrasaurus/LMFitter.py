from .imports import *
import scipy
import lmfit
from lmfit import Minimizer
#import os
from ldtk import LDPSetCreator, BoxcarFilter
from .ModelMaker import ModelMaker
from .Plotter import Plotter

class LMFitter(Talker, Writer):

    '''this class will marginalize over the provided parameters using a levenberg-marquardt minimizer'''

    def __init__(self, detrender, wavefile):
        ''' initialize the lmfitter'''
        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs.inputs
        self.cube = self.detrender.cube
        self.wavefile = wavefile
        self.savewave = self.inputs['directoryname']+self.wavefile
        
        Writer.__init__(self, self.savewave+'.txt')

        self.wavebin = np.load(self.savewave+'.npy')[()]
        subdirs = self.wavebin.keys()

        if self.wavebin['lmfitdone']:
            self.speak('lmfit already exists for wavelength bin {0}'.format(self.wavefile))

        else: 
            self.speak('running lmfit for wavelength bin {0}'.format(self.wavefile))
            self.setup()
            self.runLMFit()

    def setup(self):

        if self.inputs['ldmodel']:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        # make a list of all the aprameters you are going to fit; these are different for each wave bin so has to be don here instead of in inputs
        self.freeparamnames  = np.concatenate([self.inputs[subdir]['freeparamnames'] for subdir in self.wavebin['subdirectories']])
        self.freeparamvalues = np.concatenate([self.inputs[subdir]['freeparamvalues'] for subdir in self.wavebin['subdirectories']])
        self.freeparambounds = np.concatenate([self.inputs[subdir]['freeparambounds'] for subdir in self.wavebin['subdirectories']], 1)
        # if there is only one data set in this wavebin, nothing needs to be removed
        if len(self.wavebin['subdirectories']) > 1:
            removeinds = []
            for jointparam in self.inputs['jointparams']:
                firstdir = self.wavebin['subdirectories'][0]
                if jointparam+self.inputs[firstdir]['n'] in self.freeparamnames:
                    for subdir in self.wavebin['subdirectories'][1:]:
                        paramind = np.argwhere(np.array(self.freeparamnames) == jointparam+self.inputs[subdir]['n'])
                        paramind = np.ndarray.flatten(paramind)[0]
                        removeinds.append(paramind)
            self.freeparamnames  = np.delete(self.freeparamnames, removeinds)
            self.freeparamvalues = np.delete(self.freeparamvalues, removeinds)
            self.freeparambounds = np.delete(self.freeparambounds, removeinds, axis=1)

        self.wavebin['freeparamnames'] = self.freeparamnames

        self.lcs = np.array([self.wavebin[subdir]['lc'] for subdir in self.wavebin['subdirectories']])
        self.photnoiseest = np.array([self.wavebin[subdir]['photnoiseest'] for subdir in self.wavebin['subdirectories']])
        self.binnedok = np.array([self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']])

        self.rangeofdirectories = range(len(self.wavebin['subdirectories']))

    def runLMFit(self):

        self.speak('running first lmfit scaling by photon noise limits')#, making output txt file')

        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.freeparamvalues[n])
            if self.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.freeparambounds[0][n]
            if self.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        model = ModelMaker(self.inputs, self.wavebin, np.array(self.freeparamvalues))
        def lineareqn(params):
            return model.makemodel(np.array(list(params.valuesdict().values())))


        def residuals1(params):
            models = lineareqn(params)
            residuals = [((self.lcs[i] - models[i])/self.photnoiseest[i])[self.binnedok[i]] for i in self.rangeofdirectories]
            #for subdir in self.wavebin['subdirectories']:
                # weight by photon noise limit (expected noise); only include binnedok points in residuals - don't want masked points to determine goodness of fit
            #    residuals.append(((self.wavebin[subdir]['lc'] - model[subdir])/self.wavebin[subdir]['photnoiseest'])[self.wavebin[subdir]['binnedok']])
            return np.hstack(residuals)


        fit_kws={'epsfcn':1e-5}  # set the stepsize to something small but reasonable; withough this lmfit may have trouble perturbing values
            #, 'full_output':True, 'xtol':1e-5, 'ftol':1e-5, 'gtol':1e-5}
        self.linfit1 = lmfit.minimize(fcn=residuals1, params=lmfitparams, method='leastsq', **fit_kws)
        linfit1paramvals = list(self.linfit1.params.valuesdict().values())
        linfit1uncs = np.sqrt(np.diagonal(self.linfit1.covar))
        self.write('1st lm params:')
        [self.write('    '+self.freeparamnames[i]+'    '+str(linfit1paramvals[i])+'  +/-  '+str(linfit1uncs[i])) for i in range(len(self.freeparamnames))]

        ######### do a second fit with priors, now that you know what the initial scatter is ########
        
        self.speak('running second lmfit after clipping >{0} sigma points'.format(self.inputs['sigclip']))
 
       # median absolute deviation sigma clipping to specified sigma value from inputs
        model = ModelMaker(self.inputs, self.wavebin, np.array(linfit1paramvals))
        models = model.makemodel(np.array(linfit1paramvals))
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            resid = (self.wavebin[subdir]['lc'] - models[s])[self.wavebin[subdir]['binnedok']] # don't include masked points in residuals

            # median absolute deviation
            mad = np.median(abs(resid - np.median(resid)))
            scale = 1.4826
            data_unc = scale*mad               # scale x median absolute deviation
            
            # find indices that do not meet clipping requirement
            clippoint = (resid > (self.inputs['sigclip']*data_unc)) | (resid < (-self.inputs['sigclip']*data_unc)) # boolean array; true if point does not meet clipping requirements
            #print('clippoint', clippoint)
            
            # remake 'binnedok'
            goodinds = np.where(self.wavebin[subdir]['binnedok'])[0] # indices that were fed into the model
            clipinds = goodinds[clippoint] # just the indices that should be clipped
            self.wavebin[subdir]['binnedok'][clipinds] = False

            # save new 'binnedok' to wavebin to be used later
            self.write('clipped points for {0}: {1}'.format(subdir, clipinds))
            np.save(self.savewave, self.wavebin)
        self.binnedok = np.array([self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']])

        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.freeparamvalues[n])
            if self.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.freeparambounds[0][n]
            if self.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        def residuals2(params):
            models = lineareqn(params)
            residuals = [((self.lcs[i] - models[i])/self.photnoiseest[i])[self.binnedok[i]] for i in self.rangeofdirectories]
            #residuals = []
            #for subdir in self.wavebin['subdirectories']:
                # weight by photon noise limit (expected noise); only include binnedok points in residuals - don't want masked points to determine goodness of fit
            #    residuals.append(((self.wavebin[subdir]['lc'] - models[subdir])/self.wavebin[subdir]['photnoiseest'])[self.wavebin[subdir]['binnedok']]) 
            return np.hstack(residuals)

        self.linfit2 = lmfit.minimize(fcn=residuals2, params=lmfitparams, method='leastsq', **fit_kws)
        linfit2paramvals = list(self.linfit2.params.valuesdict().values())
        linfit2uncs = np.sqrt(np.diagonal(self.linfit2.covar))
        self.write('2nd lm params:')
        [self.write('    '+self.freeparamnames[i]+'    '+str(linfit2paramvals[i])+'  +/-  '+str(linfit2uncs[i])) for i in range(len(self.freeparamnames))]

        ######### do a third fit, now with calculated uncertainties ########
        
        self.speak('running third lmfit after calculating undertainties from the data'.format(self.inputs['sigclip']))
 
        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.freeparamvalues[n])
            if self.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.freeparambounds[0][n]
            if self.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        model = ModelMaker(self.inputs, self.wavebin, np.array(linfit2paramvals))
        models = model.makemodel(np.array(linfit2paramvals))
        data_uncs2 = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            resid = (self.wavebin[subdir]['lc'] - models[s])[self.wavebin[subdir]['binnedok']]
            data_unc = np.std(resid)
            data_uncs2.append(data_unc)
        self.write('lmfit2 data uncs: {0}'.format(data_uncs2))

        def residuals3(params):
            models = lineareqn(params)
            residuals = [((self.lcs[i] - models[i])/data_uncs2[i])[self.binnedok[i]] for i in self.rangeofdirectories]
            #for n, subdir in enumerate(self.wavebin['subdirectories']):
                # weight by calculated uncertainty; only include binnedok points in residuals - don't want masked points to determine goodness of fit
            #    residuals.append(((self.wavebin[subdir]['lc'] - models[subdir])/data_uncs2[n])[self.wavebin[subdir]['binnedok']])
            return np.hstack(residuals)

        self.linfit3 = lmfit.minimize(fcn=residuals3, params=lmfitparams, method='leastsq', **fit_kws)
        linfit3paramvals = list(self.linfit3.params.valuesdict().values())
        linfit3uncs = np.sqrt(np.diagonal(self.linfit3.covar))
        self.write('3rd lm params:')
        [self.write('    '+self.freeparamnames[i]+'    '+str(linfit3paramvals[i])+'  +/-  '+str(linfit3uncs[i])) for i in range(len(self.freeparamnames))]

        for subdir in self.wavebin['subdirectories']:
            n = str(self.inputs[subdir]['n'])
            if 'dt'+n in self.linfit3.params.keys():
                self.inputs[subdir]['t0'] = self.linfit3.params['dt'+n] + self.inputs[subdir]['toff']
                self.speak('lmfit reseting t0 parameter for {0}, transit midpoint = {1}'.format(subdir, self.inputs[subdir]['t0']))
            self.write('lmfit transit midpoint for {0}: {1}'.format(subdir, self.inputs[subdir]['t0']))

        model = ModelMaker(self.inputs, self.wavebin, np.array(linfit3paramvals))
        models = model.makemodel(np.array(linfit3paramvals))

        resid = [(self.lcs[i] - models[i])[self.binnedok[i]] for i in self.rangeofdirectories]
        #for subdir in self.wavebin['subdirectories']:
            # calculate the residuals and only include the binnedok values
        #    resid.append((self.wavebin[subdir]['lc'] - models[subdir])[self.wavebin[subdir]['binnedok']])
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('lmfit overall RMS: {0}'.format(data_unc))  # this is the same as the rms!

        # how many times the expected noise is the rms?
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            self.write('x mean expected noise for {0}: {1}'.format(subdir, np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']])))

        # make BIC,AIC calculations
        for subdir in self.wavebin['subdirectories']:
            # use linfit2 where uncertainty was taken from photon noise estimate (does not vary with fit)
            self.write('Model statistics for {0}: BIC = {1}, AIC = {2}'.format(subdir, self.linfit2.bic, self.linfit2.aic))

        self.speak('saving lmfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['lmfit'] = {}
        self.wavebin['lmfit']['freeparamnames']  = self.freeparamnames
        self.wavebin['lmfit']['freeparamvalues'] = self.freeparamvalues
        self.wavebin['lmfit']['freeparambounds'] = self.freeparambounds
        self.wavebin['lmfit']['values'] = linfit3paramvals
        try: self.wavebin['lmfit']['uncs'] = np.sqrt(np.diagonal(self.linfit3.covar))
        except(ValueError):
            self.speak('the linear fit returned no uncertainties, consider changing tranbounds values')
            return
        if not np.all(np.isfinite(np.array(self.wavebin['lmfit']['uncs']))): 
            self.speak('lmfit error: there were non-finite uncertainties')
            return
        fitmodel = {}
        batmanmodel = {}
        for s, subdir in enumerate(self.wavebin['subdirectories']): 
            fitmodel[subdir] = model.fitmodel[s]
            batmanmodel[subdir] = model.batmanmodel[s]
        self.wavebin['lmfit']['fitmodels'] = fitmodel
        self.wavebin['lmfit']['batmanmodels'] = batmanmodel
        self.wavebin['lmfitdone'] = True
        np.save(self.savewave, self.wavebin)

        plot = Plotter(self.inputs, self.cube.subcube)
        plot.lmplots(self.wavebin, [self.linfit1, self.linfit2, self.linfit3])

        self.speak('done with lmfit for wavelength bin {0}'.format(self.wavefile))

        if self.inputs['dividewhite'] and self.inputs['binlen']=='all':
            # save the transit model from the white light curve fit
            self.dividewhite = np.load(self.inputs['directoryname']+'dividewhite.npy')[()]
            self.speak('saving Twhite for later use by divide white routine')
            self.dividewhite['Twhite'] = modelobj.batmanmodel
            np.save(self.inputs['directoryname']+'dividewhite.npy', self.dividewhite)

    def limbdarkparams(self, wavestart, waveend):
        self.speak('using ldtk to derive limb darkening parameters')
        filters = BoxcarFilter('a', wavestart, waveend),     # Define passbands - Boxcar filters for transmission spectroscopy
        sc = LDPSetCreator(teff=(self.inputs['Teff'], self.inputs['Teff_unc']),             # Define your star, and the code
                           logg=(self.inputs['logg'], self.inputs['logg_unc']),             # downloads the uncached stellar 
                              z=(self.inputs['z']   , self.inputs['z_unc']),                # spectra from the Husser et al.
                        filters=filters)                      # FTP server automatically.

        ps = sc.create_profiles()                             # Create the limb darkening profiles
        if self.inputs['ldlaw'] == 'sq': 
            u , u_unc = ps.coeffs_sq(do_mc=True)                  # Estimate non-linear law coefficients
            chains = np.array(ps._samples['sq'])
            u0_array = chains[:,:,0]
            u1_array = chains[:,:,1]
        elif self.inputs['ldlaw'] == 'qd': 
            u , u_unc = ps.coeffs_qd(do_mc=True)                  # Estimate non-linear law coefficients
            chains = np.array(ps._samples['qd'])
            u0_array = chains[:,:,0]
            u1_array = chains[:,:,1]
        else: 
            self.speak('unknown limb-darkening law!')
            return
        #self.u0, self.u1 = u[0][0], u[0][1]
        #self.u0_unc, self.u1_unc = u_unc[0][0], u_unc[0][1]
        self.write('limb darkening params: '+str(u[0][0])+' +/- '+str(u_unc[0][0])+'    '+str(u[0][1])+' +/- '+str(u_unc[0][1]))

        # re-parameterize the limb darkening parameters 
        if self.inputs['ldlaw'] == 'sq':
            self.v0_array = u0_array/3. + u1_array/5.
            self.v1_array = u0_array/5. - u1_array/3.
        elif self.inputs['ldlaw'] == 'qd':
            self.v0_array = 2*u0_array + u1_array
            self.v1_array = u0_array - 2*u1_array
        self.v0, self.v1 = np.mean(self.v0_array), np.mean(self.v1_array)
        self.v0_unc, self.v1_unc = np.std(self.v0_array), np.std(self.v1_array)
        self.write('re-parameterized limb darkening params: '+str(self.v0)+' +/- '+str(self.v0_unc)+'    '+str(self.v1)+' +/- '+str(self.v1_unc))

        # save the re-parameterized limb_darkening values so that they can be recalled when making the model

        self.wavebin['ldparams'] = {}
        self.wavebin['ldparams']['v0'] = self.v0
        self.wavebin['ldparams']['v1'] = self.v1
        self.wavebin['ldparams']['v0_unc'] = self.v0_unc
        self.wavebin['ldparams']['v1_unc'] = self.v1_unc

        for subdir in self.wavebin['subdirectories']:
            self.inputs[subdir]['tranparams'][-2], self.inputs[subdir]['tranparams'][-1] = self.v0, self.v1


