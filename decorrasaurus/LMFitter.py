from .imports import *
from scipy.optimize import minimize
import lmfit
import george
from george import kernels
#import os
from ldtk import LDPSetCreator, BoxcarFilter
from .ModelMaker import ModelMaker
from .Plotter import Plotter
from copy import deepcopy

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

        self.wavebin = np.load(self.savewave+'.npy', allow_pickle=True)[()]

        if self.wavebin['lmfitdone']:
            self.speak('lmfit already exists for wavelength bin {0}'.format(self.wavefile))

        else: 
            self.speak('running lmfit for wavelength bin {0}'.format(self.wavefile))
            self.setup()
            if self.inputs['sysmodel'] == 'linear':
                self.runLMFitLinear()
            elif self.inputs['sysmodel'] == 'GP':
                self.runLMFitGP()

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

        if self.inputs['sysmodel'] == 'linear':
            # pad all of the arrays with zeros so that we can do numpy math and vectorize everything
            # it would be neater if all of this was in some other class but whatever, good enough for now
            self.lcs = self.detrender.inputs.equalizeArrays1D(np.array([self.wavebin[subdir]['lc'] for subdir in self.wavebin['subdirectories']]))
            self.photnoiseest = self.detrender.inputs.equalizeArrays1D(np.array([self.wavebin[subdir]['photnoiseest'] for subdir in self.wavebin['subdirectories']]))
            self.binnedok = self.detrender.inputs.equalizeArrays1D(np.array([self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']])).astype(bool)

        elif self.inputs['sysmodel'] == 'GP':
            self.lcs = [self.wavebin[subdir]['lc'] for subdir in self.wavebin['subdirectories']]
            self.photnoiseest = [self.wavebin[subdir]['photnoiseest'] for subdir in self.wavebin['subdirectories']]
            self.binnedok = [self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']]

        self.rangeofdirectories = range(len(self.wavebin['subdirectories']))

        firstdir = self.wavebin['subdirectories'][0]

        # if we're doing a GP then we need to set up the kernels for each night
        if self.inputs['sysmodel'] == 'GP':

            # for some weird reason you need a long string in here so that you can insert the full strings you want to. we'll delete it at the end.
            self.freeparamnames = np.append(self.freeparamnames, 'thisisaweirdstringtoputhere')

            for s, subdir in enumerate(self.wavebin['subdirectories']):

                n = self.inputs[subdir]['n']
                nregressors = len(self.inputs[subdir]['fitlabels'])

                whitenoisevalue = np.log(500e-6**2)
                whitenoiseboundlo, whitenoiseboundhi = np.log(5e-6**2), np.log(5000e-6**2)

                constantkernelvalue = np.log(np.var(self.lcs[s]))
                constantkernelboundlo, constantkernelboundhi = np.log(0.01*np.var(self.lcs[s])), np.log(100*np.var(self.lcs[s]))
                constantkernel = kernels.ConstantKernel(constantkernelvalue, ndim=nregressors, axes=range(nregressors))

                self.wavebin[subdir]['kernellabels'] = ['constantkernel']
                self.wavebin[subdir]['kernels'] = [constantkernel]

                endswithn = [freeparam.endswith(n) for freeparam in self.freeparamnames]
                endswithninds = np.array(np.where(endswithn)).flatten()
                lastn = endswithninds[-1]
                insertn = lastn+1

                self.freeparamnames = np.insert(self.freeparamnames, insertn, ['whitenoise{0}'.format(n), 'constantkernel{0}'.format(n)])
                self.freeparamvalues = np.insert(self.freeparamvalues, insertn, [whitenoisevalue, constantkernelvalue])
                self.freeparambounds = np.insert(self.freeparambounds, insertn, [[whitenoiseboundlo, constantkernelboundlo], [whitenoiseboundhi, constantkernelboundhi]], axis=1)
                #self.freeparamnames = np.append(self.freeparamnames, ['constantkernel{}'.format(n)])
                #self.freeparamvalues = np.append(self.freeparamvalues, [constantkernelvalue])
                #self.freeparambounds = np.append(self.freeparambounds, [[constantkernelboundlo], [constantkernelboundhi]], axis=1)
    
                # bump up the insertion spot to account for the whitenoise and constantkernel positions
                insertn += 2

                regressor_arrays = []

                for i, fitlabel in enumerate(self.inputs[subdir]['fitlabels']):

                    kerneltype = self.inputs[subdir]['kerneltypes'][i]

                    regressor_array = self.wavebin[subdir]['compcube'][fitlabel]
                    regressor_arrays.append(regressor_array)
                    regressor_array_range = 100*(regressor_array.max() - regressor_array.min())
                    regressor_array_spacing = np.abs(np.mean(np.diff(regressor_array)))

                    boundlo = np.log(1/regressor_array_range)
                    boundhi = np.log(1/regressor_array_spacing)
                    #if fitlabel == 'width': boundhi = np.log(7500)
                    log_inverse_length_scale = np.log(1/(5*(regressor_array.max() - regressor_array.min())))
                    metric = 1/regressor_array_range #(1/np.exp(log_inverse_length_scale))**2

                    if kerneltype == 'ExpSquaredKernel':
                        k = kernels.ExpSquaredKernel(metric, ndim=nregressors, axes=i)

                    if i == 0: fitkernel = k
                    else: fitkernel += k

                    self.wavebin[subdir]['kernellabels'].append('{0}kernel'.format(fitlabel))
                    self.wavebin[subdir]['kernels'].append(k)

                    self.freeparamnames = np.insert(self.freeparamnames, insertn, '{0}kernel{1}'.format(fitlabel, n))
                    self.freeparamvalues = np.insert(self.freeparamvalues, insertn, np.log(metric))
                    self.freeparambounds = np.insert(self.freeparambounds, insertn, [boundlo, boundhi], axis=1)

                self.wavebin[subdir]['gpkernel'] = constantkernel*fitkernel
                self.wavebin[subdir]['gpwhitenoise'] = whitenoisevalue
                self.wavebin[subdir]['gpregressor_arrays'] = np.array(regressor_arrays)

            self.freeparamnames = np.delete(self.freeparamnames, len(self.freeparamnames)-1)

            # cause need these updated in the wavebin to feed into ModelMaker
            self.wavebin['freeparamnames']  = self.freeparamnames
            self.wavebin['freeparamvalues']  = self.freeparamvalues
            self.wavebin['freeparambounds'] = self.freeparambounds

    def runLMFitLinear(self):

        self.speak('running first lmfit scaling by photon noise limits')#, making output txt file')

        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(name=name, value=self.freeparamvalues[n])
            if self.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.freeparambounds[0][n]
            if self.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        def lineareqn(params):
            return modelobj.makemodelLinear(np.array(list(params.valuesdict().values())))

        # weight first residuals by photon noise limit (expected noise); 
        # only include binnedok points in residuals - don't want masked points to determine goodness of fit
        def residuals1(params):
            models = lineareqn(params)
            return ((self.lcs - models)/self.photnoiseest)[self.binnedok]


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
        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        models = modelobj.makemodelLinear(np.array(linfit1paramvals))
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            resid = (self.lcs[s] - models[s])[self.binnedok[s]] # don't include masked points in residuals

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
        self.binnedok = self.detrender.inputs.equalizeArrays1D(np.array([self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']])).astype(bool)

        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(name=name, value=self.freeparamvalues[n])
            if self.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.freeparambounds[0][n]
            if self.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        # weight by photon noise limit (expected noise); only include binnedok points in residuals - don't want masked points to determine goodness of fit
        def residuals2(params):
            models = lineareqn(params)
            return ((self.lcs - models)/self.photnoiseest)[self.binnedok]

        self.linfit2 = lmfit.minimize(fcn=residuals2, params=lmfitparams, method='leastsq', **fit_kws)
        linfit2paramvals = list(self.linfit2.params.valuesdict().values())
        linfit2uncs = np.sqrt(np.diagonal(self.linfit2.covar))
        self.write('2nd lm params:')
        [self.write('    '+self.freeparamnames[i]+'    '+str(linfit2paramvals[i])+'  +/-  '+str(linfit2uncs[i])) for i in range(len(self.freeparamnames))]

        ######### do a third fit, now with calculated uncertainties ########
        
        self.speak('running third lmfit after calculating undertainties from the data'.format(self.inputs['sigclip']))
 
        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(name=name, value=self.freeparamvalues[n])
            if self.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.freeparambounds[0][n]
            if self.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        models = modelobj.makemodelLinear(np.array(linfit2paramvals))
        data_uncs2 = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            resid = (self.lcs[s] - models[s])[self.binnedok[s]]
            data_unc = np.std(resid)
            data_uncs2.append(data_unc)
        self.write('lmfit2 data uncs: {0}'.format(data_uncs2))
        data_uncs2 = np.array(data_uncs2)

        # weight by calculated uncertainty; only include binnedok points in residuals - don't want masked points to determine goodness of fit
        # this will make chi^2 equal to the number of data points; compare model runs using AIC and BIC built into lmfit 
        def residuals3(params):
            models = lineareqn(params)
            return ((self.lcs - models).T/data_uncs2).T[self.binnedok]

        self.linfit3 = lmfit.minimize(fcn=residuals3, params=lmfitparams, method='leastsq', **fit_kws)
        linfit3paramvals = list(self.linfit3.params.valuesdict().values())
        linfit3uncs = np.sqrt(np.diagonal(self.linfit3.covar))
        try: linfit3uncs = np.sqrt(np.diagonal(self.linfit3.covar))
        except(ValueError):
            self.speak('the linear fit returned no uncertainties, consider changing tranbounds values')
            return
        if not np.all(np.isfinite(np.array(linfit3uncs))): 
            self.speak('lmfit error: there were non-finite uncertainties')
            return
        self.write('3rd lm params:')
        [self.write('    '+self.freeparamnames[i]+'    '+str(linfit3paramvals[i])+'  +/-  '+str(linfit3uncs[i])) for i in range(len(self.freeparamnames))]

        for subdir in self.wavebin['subdirectories']:
            n = str(self.inputs[subdir]['n'])
            if 'dt'+n in self.linfit3.params.keys():
                self.inputs[subdir]['t0'] = self.linfit3.params['dt'+n] + self.inputs[subdir]['toff']
                self.speak('lmfit reseting t0 parameter for {0}, transit midpoint = {1}'.format(subdir, self.inputs[subdir]['t0']))
            self.write('lmfit transit midpoint for {0}: {1}'.format(subdir, self.inputs[subdir]['t0']))

        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        models = modelobj.makemodelLinear(np.array(linfit3paramvals))

        resid = [(self.lcs[i] - models[i])[self.binnedok[i]] for i in self.rangeofdirectories]
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
        self.wavebin['lmfit']['uncs'] = linfit3uncs

        fitmodel = {}
        batmanmodel = {}
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            if self.detrender.inputs.numberofzeros[s] == 0:
                fitmodel[subdir] = modelobj.fitmodel[s]
                batmanmodel[subdir] = modelobj.batmanmodel[s]
            else:
                fitmodel[subdir] = modelobj.fitmodel[s][:-self.detrender.inputs.numberofzeros[s]]
                batmanmodel[subdir] = modelobj.batmanmodel[s][:-self.detrender.inputs.numberofzeros[s]]
        self.wavebin['lmfit']['fitmodels'] = fitmodel
        self.wavebin['lmfit']['batmanmodels'] = batmanmodel
        self.wavebin['lmfitdone'] = True
        np.save(self.savewave, self.wavebin)

        plot = Plotter(self.inputs, self.cube.subcube)
        plot.lmplots(self.wavebin, self.linfit1, self.linfit2, self.linfit3)

        self.speak('done with lmfit for wavelength bin {0}'.format(self.wavefile))

        if self.inputs['dividewhite'] and self.inputs['binlen']=='all':
            # save the transit model from the white light curve fit
            self.dividewhite = np.load(self.inputs['directoryname']+'dividewhite.npy', allow_pickle=True)[()]
            self.speak('saving Twhite for later use by divide white routine')
            self.dividewhite['Twhite'] = modelobj.batmanmodel
            np.save(self.inputs['directoryname']+'dividewhite.npy', self.dividewhite)

    def runLMFitGP(self):

        self.speak('running 1st minimzation for the GPs')#, making output txt file')
        self.wavebin['savewave'] = self.savewave

        # clip out transit data points
        self.wavebin['fitmean'] = False
        #binnedokcopy = [deepcopy(self.wavebin[subdir]['binnedok']) for subdir in self.wavebin['subdirectories']]

        #for s, subdir in enumerate(self.wavebin['subdirectories']):
        #    intransit = (self.cube.subcube[subdir]['bjd'] > self.inputs[subdir]['t0']-(0.51*self.inputs['Tdur'])) * (self.cube.subcube[subdir]['bjd'] < self.inputs[subdir]['t0']+(0.51*self.inputs['Tdur']))
        #    self.wavebin[subdir]['binnedok'] *= np.invert(intransit)
        #self.binnedok = [self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']]
        
        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        self.gps = modelobj.makemodelGP()

        initnames = np.hstack([self.freeparamnames[modelobj.gpparaminds[i]] for i in self.rangeofdirectories])
        initvals = np.hstack([self.freeparamvalues[modelobj.gpparaminds[i]] for i in self.rangeofdirectories])
        initbounds = np.vstack([self.freeparambounds.T[modelobj.gpparaminds[i]] for i in self.rangeofdirectories])

        localgpparaminds = []
        lastparamind = 0
        for i in self.rangeofdirectories:
            localgpparaminds.append([])
            localgpparaminds[i].append(np.arange(lastparamind, lastparamind+len(modelobj.gpparaminds[i])))
            lastparamind += len(modelobj.gpparaminds[i])

        def neg_log_like(p):
            [self.gps[i].set_parameter_vector(np.array(p)[localgpparaminds[i]]) for i in self.rangeofdirectories]
            neg_log_likes = np.array([-self.gps[i].log_likelihood(self.lcs[i][self.binnedok[i]]) for i in self.rangeofdirectories])
            if np.any(neg_log_likes == np.inf):
                infinites = np.argwhere(neg_log_likes == np.inf)[0]
                neg_log_likes[infinites] = 1e25
            return np.sum(neg_log_likes)

        def grad_neg_log_like(p):
            [self.gps[i].set_parameter_vector(np.array(p)[localgpparaminds[i]]) for i in self.rangeofdirectories]
            if len(self.rangeofdirectories) > 1:
                grad_neg_log_likes = np.sum([-self.gps[i].grad_lnlikelihood(self.lcs[i][self.binnedok[i]], quiet=True) for i in self.rangeofdirectories], axis=1) # should return same number of values as gpparaminds; this will not work if diff. numbers of kernels.....
            else: grad_neg_log_likes = -self.gps[0].grad_lnlikelihood(self.lcs[0][self.binnedok[0]], quiet=True)
            return grad_neg_log_likes

        self.lmfit1 = minimize(neg_log_like, initvals, jac=grad_neg_log_like, method='L-BFGS-B', bounds=initbounds, options={'eps':1e-15, 'ftol':1e-15})
        lmfitparamvals1 = self.lmfit1.x
        lmfituncs1 = np.sqrt(np.diagonal(self.lmfit1.hess_inv*np.identity(len(self.lmfit1.x))))
        self.write('params from 1st minimization:')
        [self.write('    {0}    {1}  +/-  {2}'.format(initnames[i], lmfitparamvals1[i], lmfituncs1[i])) for i in range(len(initnames))]

        regressors = [self.wavebin[s]['gpregressor_arrays'].T[self.binnedok[i]] for i, s in enumerate(self.inputs['subdirectories'])]

        [self.gps[i].set_parameter_vector(np.array(lmfitparamvals1)[localgpparaminds[i]]) for i in self.rangeofdirectories]
        models = [self.gps[i].predict(self.lcs[i][self.binnedok[i]], regressors[i], return_cov=False) for i in self.rangeofdirectories]

        for s, subdir in enumerate(self.wavebin['subdirectories']):
            resid = (self.lcs[s][self.binnedok[s]] - models[s]) # don't include masked points in residuals
            lcnans = np.ones_like(resid)
            lcnans[:] = np.nan
            residnans = np.ones_like(resid)
            residnans[:] = np.nan

            # median absolute deviation
            mad = np.median(abs(resid - np.median(resid)))
            scale = 1.4826
            data_unc = scale*mad               # scale x median absolute deviation
            
            # find indices that do not meet clipping requirement
            clippoint = (resid > (self.inputs['sigclip']*data_unc)) | (resid < (-self.inputs['sigclip']*data_unc)) # boolean array; true if point does not meet clipping requirements
            lcnans[clippoint] = self.lcs[s][self.binnedok[s]][clippoint]
            residnans[clippoint] = resid[clippoint]
            self.wavebin[subdir]['lcclipped'] = lcnans
            self.wavebin[subdir]['residclipped'] = residnans

            # remake 'binnedok'
            goodinds = np.where(self.wavebin[subdir]['binnedok'])[0] # indices that were fed into the model
            clipinds = goodinds[clippoint] # just the indices that should be clipped
            self.wavebin[subdir]['binnedokunclipped'] = deepcopy(self.wavebin[subdir]['binnedok'])
            self.wavebin[subdir]['binnedok'][clipinds] = False
            self.binnedok[s][clipinds] = False

            # save new 'binnedok' to wavebin to be used later
            self.write('clipped points for {0}: {1}'.format(subdir, clipinds))
        np.save(self.savewave, self.wavebin)

        self.speak('running 2nd minimzation for the GPs')#, making output txt file')
        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        self.gps = modelobj.makemodelGP()

        self.lmfit2 = minimize(neg_log_like, lmfitparamvals1, jac=grad_neg_log_like, method='L-BFGS-B', bounds=initbounds, options={'eps':1e-15, 'ftol':1e-15})
        lmfitparamvals2 = self.lmfit2.x
        lmfituncs2 = np.sqrt(np.diagonal(self.lmfit2.hess_inv*np.identity(len(self.lmfit2.x))))
        self.write('params from 1st minimization:')
        [self.write('    {0}    {1}  +/-  {2}'.format(initnames[i], lmfitparamvals2[i], lmfituncs2[i])) for i in range(len(initnames))]

        regressors = [self.wavebin[s]['gpregressor_arrays'].T[self.binnedok[i]] for i, s in enumerate(self.inputs['subdirectories'])]

        [self.gps[i].set_parameter_vector(np.array(lmfitparamvals2)[localgpparaminds[i]]) for i in self.rangeofdirectories]
        models = [self.gps[i].predict(self.lcs[i][self.binnedok[i]], regressors[i], return_cov=False) for i in self.rangeofdirectories]
        
        for s, subdir in enumerate(self.inputs['subdirectories']):
            self.wavebin[subdir]['kernelmus'] = []
            pred, pred_var = self.gps[s].predict(self.lcs[s][self.binnedok[s]], regressors[s], return_var=True)
            self.wavebin[subdir]['model_var'] = pred_var
            
            for kernel in self.wavebin[subdir]['kernels']:
                mu = self.gps[s].predict(self.lcs[s][self.binnedok[s]], regressors[s], return_cov=False, kernel=kernel)
                self.wavebin[subdir]['kernelmus'].append(mu)

        resid = [(self.lcs[i][self.binnedok[i]] - models[i]) for i in self.rangeofdirectories]
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('lmfit overall RMS: {0}'.format(data_unc))  # this is the same as the rms!

        # how many times the expected noise is the rms?
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            self.write('x mean expected noise for {0}: {1}'.format(subdir, np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']])))

        finallmfitvalues = np.array(self.freeparamvalues)
        for i in self.rangeofdirectories: finallmfitvalues[modelobj.gpparaminds[i]] = lmfitparamvals2[localgpparaminds[i]]

        self.speak('saving lmfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['lmfit'] = {}
        self.wavebin['lmfit']['freeparamnames']  = self.freeparamnames
        self.wavebin['lmfit']['freeparamvalues'] = self.freeparamvalues
        self.wavebin['lmfit']['freeparambounds'] = self.freeparambounds
        self.wavebin['lmfit']['values'] = finallmfitvalues
        self.wavebin['lmfit']['uncs'] = np.ones(len(self.freeparamvalues))

        fitmodel = {}
        batmanmodel = {}
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            batmanmodel[subdir] = np.ones_like(self.lcs[s][self.binnedok[s]])
            fitmodel[subdir] = models[s]/batmanmodel[subdir]

        self.wavebin['lmfit']['fitmodels'] = fitmodel
        self.wavebin['lmfit']['batmanmodels'] = batmanmodel
        self.wavebin['lmfitdone'] = True

        # do a quick save in case there's an error in the plotting
        np.save(self.savewave, self.wavebin)

        plot = Plotter(self.inputs, self.cube.subcube)
        plot.lmplots(self.wavebin)

        # revert to binned ok that does not eliminate transit
        #for s, subdir in enumerate(self.wavebin['subdirectories']): self.wavebin[subdir]['binnedok'] = binnedokcopy[s]

        # re-save the wavebin to reflect the correct binnedok arrays
        np.save(self.savewave, self.wavebin)


        self.speak('done with lmfit for wavelength bin {0}'.format(self.wavefile))

        if self.inputs['dividewhite'] and self.inputs['binlen']=='all':
            # save the transit model from the white light curve fit
            self.dividewhite = np.load(self.inputs['directoryname']+'dividewhite.npy', allow_pickle=True)[()]
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
            u , u_unc = ps.coeffs_qd(do_mc=True)                  # Estimate quadratic law with triangle method from Kipping+ (2013)
            chains = np.array(ps._samples['qd'])
            u0_array = chains[:,:,0]
            u1_array = chains[:,:,1]
        elif self.inputs['ldlaw'] == 'lg':                        # logarithmic law (Espinoza+ 2016) This law does not exist in ldtk, but that's okay we only need the boudns
            u, u_unc = ps.coeffs_qd(do_mc=True)
            chains = np.array(ps._samples['qd'])
            u0_array = chains[:,:,0]
            u1_array = chains[:,:,1]
        else: 
            self.speak('unknown limb-darkening law!')
            return
        self.write('limb darkening params: '+str(u[0][0])+' +/- '+str(u_unc[0][0])+'    '+str(u[0][1])+' +/- '+str(u_unc[0][1]))

        # re-parameterize the limb darkening parameters according to Kipping+ (2013)
        if self.inputs['ldlaw'] == 'sq':
            self.q0_array = (u0_array + u1_array)**2
            self.q1_array = u1_array/(2*(u0_array + u1_array))
        elif self.inputs['ldlaw'] == 'qd':
            self.q0_array = (u0_array + u1_array)**2
            self.q1_array = u0_array/(2*(u0_array + u1_array))
        elif self.inputs['ldlaw'] == 'lg':
            self.q0_array = (1 - u1_array)**2
            self.q1_array = (1 - u0_array)/(1 - u1_array)
        self.q0, self.q1 = np.mean(self.q0_array), np.mean(self.q1_array)
        self.q0_unc, self.q1_unc = np.std(self.q0_array), np.std(self.q1_array)
        self.write('re-parameterized limb darkening params: '+str(self.q0)+' +/- '+str(self.q0_unc)+'    '+str(self.q1)+' +/- '+str(self.q1_unc))

        # save the re-parameterized limb_darkening values so that they can be recalled when making the model
        self.wavebin['ldparams'] = {}
        self.wavebin['ldparams']['q0'] = self.q0
        self.wavebin['ldparams']['q1'] = self.q1
        self.wavebin['ldparams']['q0_unc'] = self.q0_unc
        self.wavebin['ldparams']['q1_unc'] = self.q1_unc

        for subdir in self.wavebin['subdirectories']:
            self.inputs[subdir]['tranparams'][-2], self.inputs[subdir]['tranparams'][-1] = self.q0, self.q1
            self.inputs[subdir]['tranbounds'][0][-2], self.inputs[subdir]['tranbounds'][0][-1] = max(self.q0-5*self.q0_unc, 0.01), max(self.q1-5*self.q1_unc, 0.01)
            self.inputs[subdir]['tranbounds'][1][-2], self.inputs[subdir]['tranbounds'][1][-1] = min(self.q0+5*self.q0_unc, 0.99), min(self.q1+5*self.q1_unc, 0.99)
            

