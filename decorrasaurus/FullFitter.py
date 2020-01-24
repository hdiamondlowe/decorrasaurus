from .imports import *
import emcee
import dynesty
from dynesty.dynamicsampler import stopping_function, weight_function
from dynesty.plotting import _quantile
from dynesty import utils as dyfunc
import lmfit
import sys
from ldtk import LDPSetCreator, BoxcarFilter
from .ModelMaker import ModelMaker
from .CubeReader import CubeReader
from .Plotter import Plotter
from multiprocessing import Pool
from scipy import stats
import scipy.interpolate as interpolate
#import emceehelper as mc


class FullFitter(Talker, Writer):

    '''this class will marginalize over the provided parameters using an mcmc (emcee) or a dynamic nested sampler (dynesty)'''
    '''WARNING: mcmc depricated; use dynesty instead'''

    def __init__(self, detrender, wavefile):
        ''' initialize the lmfitter'''
        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs.inputs
        self.subcube = self.detrender.cube.subcube
        self.wavefile = wavefile
        self.savewave = self.inputs['directoryname']+self.wavefile
        
        Writer.__init__(self, self.savewave+'.txt')

        self.wavebin = np.load(self.savewave+'.npy', allow_pickle=True)[()]
        if self.wavebin['mcfitdone']:
            self.speak('mcfit already exists for wavelength bin {0}'.format(self.wavefile))
            if self.inputs['samplecode'] == 'emcee':
                if self.wavebin['mcfit']['chain'].shape[1] == self.inputs.nsteps:
                    self.speak('mcfit has completed the number of steps')
                elif self.wavebin['mcfit']['chain'].shape[1] < self.inputs.nsteps:
                    self.speak('extending mcfit for more steps')
                    self.runFullFit_emcee()
        else: 
            self.speak('running mcfit for wavelength bin {0}'.format(self.wavefile))
            self.setup()
            if self.inputs['samplecode']   == 'emcee':   self.runFullFit_emcee()  # this option probably doesn't actually work anymore
            elif self.inputs['sysmodel'] == 'linear' and self.inputs['samplecode'] == 'dynesty': self.runFullFitLinear_dynesty()
            elif self.inputs['sysmodel'] == 'GP' and self.inputs['samplecode'] == 'dynesty': self.runFullFitGP_dynesty()

    def setup(self):

        self.rangeofdirectories = range(len(self.wavebin['subdirectories']))

        # get the subdirecotry index of the first subdirectory used in this wavebin
        firstdir = self.wavebin['subdirectories'][0]
        self.firstn = self.inputs[firstdir]['n']

        # limb darkening parameters were fixed in the Levenberg-Marquardt fits but now we want to fit for them
        self.freeparamnames = self.wavebin['lmfit']['freeparamnames']
        self.freeparamvalues = self.wavebin['lmfit']['values']
        self.freeparambounds = self.wavebin['lmfit']['freeparambounds']

        # append u0+firstn to the free paramnames
        self.freeparamnames = np.append(self.freeparamnames, 'u0'+self.firstn)
        self.freeparamvalues = np.append(self.freeparamvalues, self.wavebin['ldparams']['q0'])
        # the 'q' versions of the limb-darkening parameters are reparameterized according to Kipping+ (2013) such that they can be uniformly sampled from [0,1]
        self.freeparambounds = np.append(self.freeparambounds, [[0], [1]], axis=1)
        # append u1+firstn to the free paramnames
        self.freeparamnames = np.append(self.freeparamnames, 'u1'+self.firstn)
        self.freeparamvalues = np.append(self.freeparamvalues, self.wavebin['ldparams']['q1'])
        self.freeparambounds = np.append(self.freeparambounds, [[0], [1]], axis=1)
                
        # add these scaling parameters to fit for; ideally they would be 1 but likely they will turn out slightly higher
        if self.inputs['sysmodel'] == 'linear':
            for subdir in self.wavebin['subdirectories']:
                n = self.inputs[subdir]['n']
                self.freeparamnames = np.append(self.freeparamnames, 's'+n)
                self.freeparamvalues = np.append(self.freeparamvalues, 1)
                self.freeparambounds = np.append(self.freeparambounds, [[0.01], [10.]], axis=1)

        self.wavebin['freeparamnames'] = self.freeparamnames

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.freeparambounds[1]]

        if self.inputs['sysmodel'] == 'linear':
            for i, name in enumerate(self.wavebin['lmfit']['freeparamnames']):
                if self.mcmcbounds[0][i] == True:
                    self.mcmcbounds[0][i] = self.wavebin['lmfit']['values'][i] - self.wavebin['lmfit']['uncs'][i]*10.
                if self.mcmcbounds[1][i] == True: 
                    self.mcmcbounds[1][i] = self.wavebin['lmfit']['values'][i] + self.wavebin['lmfit']['uncs'][i]*10.
        #elif self.inputs['sysmodel'] == 'GP':
        #    for i, name in enumerate(self.wavebin['lmfit']['freeparamnames']):
        #        if self.mcmcbounds[0][i] == None:
        #            self.mcmcbounds[0][i] = self.wavebin['lmfit']['values'][i] - self.wavebin['lmfit']['uncs'][i]*5.
        #        if self.mcmcbounds[1][i] == None: 
        #            self.mcmcbounds[1][i] = self.wavebin['lmfit']['values'][i] + self.wavebin['lmfit']['uncs'][i]*5.

            self.mcmcbounds = np.array(self.mcmcbounds)

        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        if self.inputs['sysmodel'] == 'linear':
            # pad the uneven lists so that they can be numpy arrays
            self.lcs = self.detrender.inputs.equalizeArrays1D(np.array([self.wavebin[subdir]['lc'] for subdir in self.wavebin['subdirectories']]))
            self.photnoiseest = self.detrender.inputs.equalizeArrays1D(np.array([self.wavebin[subdir]['photnoiseest'] for subdir in self.wavebin['subdirectories']]))
            self.binnedok = self.detrender.inputs.equalizeArrays1D(np.array([self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']])).astype(bool)
            self.sinds = np.array([np.argwhere(np.array(self.freeparamnames) == 's{}'.format(s))[0][0] for s, subdir in enumerate(self.wavebin['subdirectories'])])
            self.numpoints = np.array([len(self.lcs[i][self.binnedok[i]]) for i in self.rangeofdirectories])

        elif self.inputs['sysmodel'] == 'GP':
            self.lcs = [self.wavebin[subdir]['lc'] for subdir in self.wavebin['subdirectories']]
            self.photnoiseest = [self.wavebin[subdir]['photnoiseest'] for subdir in self.wavebin['subdirectories']]
            self.binnedok = [self.wavebin[subdir]['binnedok'] for subdir in self.wavebin['subdirectories']]

    def runFullFit_emcee(self):

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.inputs.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.inputs.freeparambounds[1]]

        for u in range(len(self.inputs.freeparamnames)):
            if type(self.mcmcbounds[0][u]) == bool and self.mcmcbounds[0][u] == True: self.mcmcbounds[0][u] = self.wavebin['lmfit']['values'][u]-self.wavebin['lmfit']['uncs'][u]*100.
            if type(self.mcmcbounds[1][u]) == bool and self.mcmcbounds[1][u] == True: self.mcmcbounds[1][u] = self.wavebin['lmfit']['values'][u]+self.wavebin['lmfit']['uncs'][u]*100.
        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.inputs.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        #this is a hack to try to make emcee picklable, not sure this helps at all...
        def makemodellocal(inputs, wavebin, p):
            modelobj = ModelMaker(inputs, wavebin, p)
            return modelobj.makemodel()

        def lnlike(p, lcb, inputs, wavebin):
            models = makemodellocal(inputs, wavebin, p)
            residuals = []
            for n, night in enumerate(self.inputs.nightname):
                residuals.append((self.wavebin['lc'][n] - models[n])[self.wavebin['binnedok'][n]])
            residuals = np.hstack(residuals)
            data_unc = (np.std(residuals))**2
            return -0.5*np.sum((residuals)**2/data_unc + np.log(2.*np.pi*data_unc))

        def lnprior(p):
            for i in range(len(p)):
                if not (self.mcmcbounds[0][i] <= p[i] <= self.mcmcbounds[1][i]):
                    return -np.inf
            return 0.0

        llvalues = []
        def lnprobfcn(p, lcb, inputs, wavebin):
            lp = lnprior(p)
            if not np.isfinite(lp):
                return -np.inf
            ll = lnlike(p, lcb, inputs, wavebin)
            llvalues.append(ll)
            return lp + ll

        ndim = len(self.inputs.freeparamnames)
        try: 
            if 'chain' in self.wavebin['mcfit'].keys():
                # want to run the remaining steps
                self.speak('starting mcfit from last step in already existing chain')
        except(KeyError):
            self.wavebin['mcfit'] = {}
            pos = [self.wavebin['lmfit']['values'] + self.wavebin['lmfit']['uncs']*1e-4*np.random.randn(ndim) for i in range(self.inputs.nwalkers)]
            self.speak('initiating emcee')
            self.sampler = emcee.EnsembleSampler(self.inputs.nwalkers, ndim, lnprobfcn, args=([self.wavebin['lc'], self.inputs, self.wavebin]))
            self.speak('running emcee')
            self.sampler.run_mcmc(pos, 100)
            self.speak('creating chain in mcfit dictionary')
            self.wavebin['mcfit']['chain'] = self.sampler.chain
            #np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        while self.wavebin['mcfit']['chain'].shape[1] < self.inputs.nsteps:
            pos = self.wavebin['mcfit']['chain'][:,-1,:]
            self.sampler = emcee.EnsembleSampler(self.inputs.nwalkers, ndim, lnprobfcn, args=([self.wavebin['lc'], self.inputs, self.wavebin]))
            for i, result in enumerate(self.sampler.sample(pos, iterations=1000)):
                if (i+1) == 1000:
                    self.wavebin['mcfit']['chain'] = np.append(self.wavebin['mcfit']['chain'], self.sampler.chain, axis=1)
                    self.speak('{0:5.1%}'.format(float(self.wavebin['mcfit']['chain'].shape[1])/self.inputs.nsteps))
                    self.speak('updating chain at step {0}'.format(self.wavebin['mcfit']['chain'].shape[1]))
                    try: np.save(self.savewave, self.wavebin)
                    except: 
                        self.speak('THERE WAS AN ERROR SAVING THE WAVEBIN')
                        return
                    try:
                        self.speak('autocorrelation array: {0}'.format(self.sampler.acor))
                    except: pass

        #self.speak('sending parameters to emcee helper')
        #self.sampler, llvalues = mc.runemcee(self, self.inputs, self.cube, self.wavebin)

        self.samples = self.wavebin['mcfit']['chain'][:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.freeparamnames)))
        self.mcparams = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(self.samples, [16, 50, 84], axis=0))))

        # This is a goofy hack: I am using the lmfit Parameters class as storage for the mcmc values and uncertainties from emcee

        self.write('mcmc acceptance: '+str(np.median(self.sampler.acceptance_fraction)))

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.inputs.freeparamnames[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.inputs.freeparamnames))]

        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                ind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                self.inputs.t0 = self.mcparams[ind] + self.inputs.toff
                self.speak('mcfit reseting t0 parameter for {0}, transit midpoint = {1}'.format(night, self.inputs.t0[n]))
            self.write('mcfit transit midpoint for {0}: {1}'.format(night, self.inputs.t0[n]))

        #calculate rms from mcfit
        modelobj = ModelMaker(self.inputs, self.wavebin, self.mcparams[:,0])
        models = modelobj.makemodel()
        resid = []
        for n in range(len(self.inputs.nightname)):
            resid.append(self.wavebin['lc'][n] - models[n])
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('mcfit RMS: '+str(np.sqrt(np.sum(allresid**2)/len(allresid))))

        # how many times the expected noise is the rms?
        for n, night in enumerate(self.inputs.nightname):
            self.write('x mean expected noise for {0}: {1}'.format(night, np.std(resid[n])/np.mean(self.wavebin['photnoiseest'][n])))
        self.write('x median mean expected noise for fit: {0}'.format(np.median([np.std(resid[n])/np.mean(self.wavebin['photnoiseest'][n]) for n in range(len(self.inputs.nightname))])))


        self.speak('saving mcfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['mcfit']['values'] = self.mcparams
        np.save(self.savewave, self.wavebin)

        plot = Plotter(self.inputs, self.subcube)
        plot.fullplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))

    def runFullFitLinear_dynesty(self):

        # rescaling uncertainties as a free parameter during the fit (Berta, et al. 2011, references therein)
        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        def lnlike(p):
            models = modelobj.makemodelLinear(p)
            # likelihood function; follow Berta+ (2012) for scaling
            logl = -self.numpoints * np.log(p[self.sinds]) - 0.5*(1./(p[self.sinds]**2))*np.nansum((((self.lcs - models)/self.photnoiseest)*self.binnedok)**2, axis=1)
            return np.sum(logl)

        span = self.mcmcbounds[1] - self.mcmcbounds[0]

        # inverse transform sampling    
        # calculate inverse cdf (ppf) of gaussian priors on u0 and u1; interpolations can be used to assign values in prior transform
        v = np.linspace(0, 1, 100000)
        ppf_u0 = stats.norm.ppf(v, loc=self.wavebin['ldparams']['q0'], scale=self.wavebin['ldparams']['q0_unc'])
        ppf_func_u0 = interpolate.interp1d(v, ppf_u0)
        ppf_u1 = stats.norm.ppf(v, loc=self.wavebin['ldparams']['q1'], scale=self.wavebin['ldparams']['q1_unc'])
        ppf_func_u1 = interpolate.interp1d(v, ppf_u1)
        u0ind = np.argwhere(np.array(self.freeparamnames) == 'u0'+self.firstn)[0][0]
        u1ind = np.argwhere(np.array(self.freeparamnames) == 'u1'+self.firstn)[0][0]
        def ptform(p):

            x = np.array(p)
            x = x*span + self.mcmcbounds[0]
            
            if x[u0ind] < 0.0001: x[u0ind] = 0.0001     # this prevents trying to interpolate a value that is beyond the bounds of the interpolation
            if x[u0ind] > .9999: x[u0ind] = .9999
            else: x[u0ind] = ppf_func_u0(x[u0ind])

            if x[u1ind] < 0.0001: x[u1ind] = 0.0001     # this prevents trying to interpolate a value that is beyond the bounds of the interpolation
            if x[u1ind] > .9999: x[u1ind] = .9999
            else: x[u1ind] = ppf_func_u1(x[u1ind])

            return x

        ndim = len(self.freeparamnames)

        self.speak('running dynesty')

        if ndim > 25: # use special inputs that will make run more efficient
            self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, bound='multi', sample='slice')
            self.dsampler.run_nested(nlive_init=int(5*ndim), nlive_batch=int(5*ndim), wt_kwargs={'pfrac': 1.0}) # place 100% of the weight on the posterior, don't sample the evidence
        else: # use defaults
            self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, sample='slice')
            self.dsampler.run_nested(wt_kwargs={'pfrac': 1.0})

        results = self.dsampler.results
        samples = results.samples
        # get the best fit +/- 1sigma uncertainties for each parameter; need to weight by the scaled logwt so that the "burn-in" samples are down-weighted
        quantiles = [dyfunc.quantile(samps, [.16, .5, .84], weights=np.exp(results['logwt']-results['logwt'][-1])) for samps in samples.T]
        self.mcparams = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), quantiles))) # had to add list() for python3

        self.speak('saving mcfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['mcfit'] = {}
        self.wavebin['mcfit']['results'] = self.dsampler.results
        self.wavebin['mcfit']['values'] = self.mcparams[:,0]
        self.wavebin['mcfit']['uncs'] = self.mcparams[:,1:]
        np.save(self.savewave, self.wavebin)

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.freeparamnames[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.freeparamnames))]

        #calculate rms from mcfit
        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        models = modelobj.makemodelLinear(self.mcparams[:,0])
        resid = [(self.lcs[i] - models[i])[self.binnedok[i]] for i in self.rangeofdirectories]
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('mcfit overall RMS: '+str(data_unc))

        # how many times the expected noise is the rms?
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            self.write('x mean expected noise for {0}: {1}'.format(subdir, np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']])))
        self.write('x median mean expected noise for joint fit: {0}'.format(np.median([np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']]) for n in range(len(self.wavebin['subdirectories']))])))

        # sort the fit and batman models into their subdirectories; remove the padded values from the ends of each
        fitmodel = {}
        batmanmodel = {}
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            if self.detrender.inputs.numberofzeros[s] == 0:
                fitmodel[subdir] = modelobj.fitmodel[s]
                batmanmodel[subdir] = modelobj.batmanmodel[s]
            else:
                fitmodel[subdir] = modelobj.fitmodel[s][:-self.detrender.inputs.numberofzeros[s]]
                batmanmodel[subdir] = modelobj.batmanmodel[s][:-self.detrender.inputs.numberofzeros[s]]
        self.wavebin['mcfit']['fitmodels'] = fitmodel
        self.wavebin['mcfit']['batmanmodels'] = batmanmodel
        self.wavebin['mcfit']['freeparamnames'] = self.freeparamnames
        self.wavebin['mcfit']['freeparamvalues'] = self.freeparamvalues
        self.wavebin['mcfit']['freeparambounds'] = self.freeparambounds
        self.wavebin['mcfitdone'] = True
        np.save(self.savewave, self.wavebin)

        plot = Plotter(self.inputs, self.subcube)
        plot.fullplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))

    def runFullFitGP_dynesty(self):

        # rescaling uncertainties as a free parameter during the fit (Berta, et al. 2011, references therein)
        modelobj = ModelMaker(self.detrender.inputs, self.wavebin)
        self.gps = modelobj.makemodelGP(self.freeparamvalues)
        self.mcmcbounds = np.concatenate([np.array(self.gps[i].get_parameter_bounds()) for i in self.rangeofdirectories]).T

        def lnlike(p):  
            [self.gps[i].set_parameter_vector(np.array(p)[modelobj.allparaminds[i]]) for i in self.rangeofdirectories]
            return np.sum([self.gps[i].log_likelihood(self.lcs[i][self.binnedok[i]]) for i in self.rangeofdirectories])

        span = self.mcmcbounds[1] - self.mcmcbounds[0]
        print(self.freeparamvalues)
        print(self.mcmcbounds)

        # need to know the indices that correspond to the kernel parameters; these should be log uniform priors
        kernelinds = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            n = self.inputs[subdir]['n']
            freekernelnames = [label+n for label in self.inputs[subdir]['kernellabels']]
            kernelinds.append([self.inputs[subdir]['freeparamnames'].index(x) for x in freekernelnames])
        print(kernelinds)
        kernelinds = np.array(kernelinds)
        expboundslo = np.exp(self.mcmcbounds[0])
        expspan = np.exp(self.mcmcbounds[1]) - np.exp(self.mcmcbounds[0])

        # need to know limb-darkening indices; these should be Gaussian priors that are bounded between 0 and 1 (from Kipping+2013)
        # calculate inverse cdf (ppf) of gaussian priors on u0 and u1; interpolations can be used to assign values in prior transform
        v = np.linspace(0, 1, 100000)
        ppf_u0 = stats.norm.ppf(v, loc=self.wavebin['ldparams']['q0'], scale=self.wavebin['ldparams']['q0_unc'])
        ppf_func_u0 = interpolate.interp1d(v, ppf_u0)
        ppf_u1 = stats.norm.ppf(v, loc=self.wavebin['ldparams']['q1'], scale=self.wavebin['ldparams']['q1_unc'])
        ppf_func_u1 = interpolate.interp1d(v, ppf_u1)
        u0ind = np.argwhere(np.array(self.freeparamnames) == 'u0'+self.firstn)[0][0]
        u1ind = np.argwhere(np.array(self.freeparamnames) == 'u1'+self.firstn)[0][0]

        def ptform(p):

            x = np.array(p)
            x = x*span + self.mcmcbounds[0]

            loguniformdist = [np.log(p[i]*expspan[i] + expboundslo[i]) for i in kernelinds]
            x[kernelinds] = loguniformdist

            if x[u0ind] < 0.0001: x[u0ind] = 0.0001     # this prevents trying to interpolate a value that is beyond the bounds of the interpolation
            if x[u0ind] > .9999: x[u0ind] = .9999
            else: x[u0ind] = ppf_func_u0(x[u0ind])

            if x[u1ind] < 0.0001: x[u1ind] = 0.0001     # this prevents trying to interpolate a value that is beyond the bounds of the interpolation
            if x[u1ind] > .9999: x[u1ind] = .9999
            else: x[u1ind] = ppf_func_u1(x[u1ind])

            return x

        ndim = len(self.freeparamnames)

        self.speak('running dynesty')

        if ndim > 25: # use special inputs that will make run more efficient
            self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, bound='multi', sample='slice')
            self.dsampler.run_nested(nlive_init=int(5*ndim), nlive_batch=int(5*ndim), wt_kwargs={'pfrac': 1.0}) # place 100% of the weight on the posterior, don't sample the evidence
        else: # use defaults
            self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, sample='slice')
            self.dsampler.run_nested(wt_kwargs={'pfrac': 1.0})

        results = self.dsampler.results
        samples = results.samples
        # get the best fit +/- 1sigma uncertainties for each parameter; need to weight by the scaled logwt so that the "burn-in" samples are down-weighted
        quantiles = [dyfunc.quantile(samps, [.16, .5, .84], weights=np.exp(results['logwt']-results['logwt'][-1])) for samps in samples.T]
        self.mcparams = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), quantiles))) # had to add list() for python3

        self.speak('saving mcfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['mcfit'] = {}
        self.wavebin['mcfit']['results'] = results
        self.wavebin['mcfit']['values'] = self.mcparams[:,0]
        self.wavebin['mcfit']['uncs'] = self.mcparams[:,1:]
        np.save(self.savewave, self.wavebin)

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.freeparamnames[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.freeparamnames))]

        #calculate rms from mcfit
        [self.gps[i].set_parameter_vector(np.array(self.mcparams[:,0])[modelobj.allparaminds[i]]) for i in self.rangeofdirectories]
        models = [self.gps[i].predict(self.lcs[i][self.binnedok[i]], modelobj.times[i], return_cov=False) for i in self.rangeofdirectories]

        resid = [(self.lcs[i][self.binnedok[i]] - models[i]) for i in self.rangeofdirectories]
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('lmfit overall RMS: {0}'.format(data_unc))  # this is the same as the rms!

        # how many times the expected noise is the rms?
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            self.write('x mean expected noise for {0}: {1}'.format(subdir, np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']])))
        self.write('x median mean expected noise for joint fit: {0}'.format(np.median([np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']]) for n in range(len(self.wavebin['subdirectories']))])))

        # sort the fit and batman models into their subdirectories; remove the padded values from the ends of each
        fitmodel = {}
        batmanmodel = {}
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            batmanmodel[subdir] = self.gps[s].mean.get_value(modelobj.times[s])
            fitmodel[subdir] = models[s]/self.gps[s].mean.get_value(modelobj.times[s])

        self.wavebin['mcfit']['fitmodels'] = fitmodel
        self.wavebin['mcfit']['batmanmodels'] = batmanmodel
        self.wavebin['mcfit']['freeparamnames'] = self.freeparamnames
        self.wavebin['mcfit']['freeparamvalues'] = self.freeparamvalues
        self.wavebin['mcfit']['freeparambounds'] = self.freeparambounds
        self.wavebin['mcfitdone'] = True
        np.save(self.savewave, self.wavebin)

        plot = Plotter(self.inputs, self.subcube)
        plot.fullplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))


