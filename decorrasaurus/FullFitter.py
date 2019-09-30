from .imports import *
import emcee
import dynesty
from dynesty.dynamicsampler import stopping_function, weight_function
from dynesty.plotting import _quantile
import lmfit
import sys
from ldtk import LDPSetCreator, BoxcarFilter
from emcee.utils import MPIPool
from .ModelMaker import ModelMaker
from .CubeReader import CubeReader
from .Plotter import Plotter
import multiprocessing
from scipy import stats
import scipy.interpolate as interpolate
#import emceehelper as mc

class FullFitter(Talker, Writer):

    '''this class will marginalize over the provided parameters using an mcmc'''

    def __init__(self, detrender, wavefile):
        ''' initialize the lmfitter'''
        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs.inputs
        self.subcube = self.detrender.cube.subcube
        self.wavefile = wavefile
        self.savewave = self.inputs['directoryname']+self.wavefile
        
        Writer.__init__(self, self.savewave+'.txt')

        self.wavebin = np.load(self.savewave+'.npy')[()]
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
            if self.inputs['samplecode']   == 'emcee':   self.runFullFit_emcee()
            elif self.inputs['samplecode'] == 'dynesty': self.runFullFit_dynesty()

    def setup(self):

        # get the subdirecotry index of the first subdirectory used in this wavebin
        firstdir = self.wavebin['subdirectories'][0]
        self.firstn = self.inputs[firstdir]['n']

        # limb darkening parameters were fixed in the Levenberg-Marquardt fits but now we want to fit for them
        self.freeparamnames = self.wavebin['lmfit']['freeparamnames']
        self.freeparamvalues = self.wavebin['lmfit']['freeparamvalues']
        self.freeparambounds = self.wavebin['lmfit']['freeparambounds']

        # append u0+firstn to the free paramnames
        self.freeparamnames = np.append(self.freeparamnames, 'u0'+self.firstn)
        self.freeparamvalues = np.append(self.freeparamvalues, self.wavebin['ldparams']['v0'])
        # these additional bounds never acutally get used - Gaussian priors get used in the prior transform function; these are just place holders
        self.freeparambounds = np.append(self.freeparambounds, [[0], [1]], axis=1)
        # append u1+firstn to the free paramnames
        self.freeparamnames = np.append(self.freeparamnames, 'u1'+self.firstn)
        self.freeparamvalues = np.append(self.freeparamvalues, self.wavebin['ldparams']['v1'])
        self.freeparambounds = np.append(self.freeparambounds, [[0], [1]], axis=1)
                
        # add these scaling parameters to fit for; ideally they would be 1 but likely they will turn out slightly higher
        for subdir in self.wavebin['subdirectories']:
            n = self.inputs[subdir]['n']
            self.freeparamnames = np.append(self.freeparamnames, 's'+n)
            self.freeparamvalues = np.append(self.freeparamvalues, 1)
            self.freeparambounds = np.append(self.freeparambounds, [[0.01], [10.]], axis=1)

        self.wavebin['freeparamnames'] = self.freeparamnames

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.freeparambounds[1]]

        for i, name in enumerate(self.wavebin['lmfit']['freeparamnames']):
            if self.mcmcbounds[0][i] == True:
                self.mcmcbounds[0][i] = self.wavebin['lmfit']['values'][i] - self.wavebin['lmfit']['uncs'][i]*10.
            if self.mcmcbounds[1][i] == True: 
                self.mcmcbounds[1][i] = self.wavebin['lmfit']['values'][i] + self.wavebin['lmfit']['uncs'][i]*10.

        self.mcmcbounds = np.array(self.mcmcbounds)

        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        self.sinds = []
        self.photnoiseests = []
        self.binnedoks = []
        self.lcs = []
        for s, subdir in enumerate(self.wavebin['subdirectories']):
            self.sinds.append(np.argwhere(np.array(self.freeparamnames) == 's'+n)[0][0])
            self.photnoiseests.append(self.wavebin[subdir]['photnoiseest'])
            self.binnedoks.append(self.wavebin[subdir]['binnedok'])
            self.lcs.append(self.wavebin[subdir]['lc'])

        self.rangeofdirectories = range(len(self.wavebin['subdirectories']))

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

    def runFullFit_dynesty(self):

        # rescaling uncertainties as a free parameter during the fit (Berta, et al. 2011, references therein)
        modelobj = ModelMaker(self.inputs, self.wavebin)
        def lnlike(p):
            # test what the priors look like on their own
            #rpind = int(np.where(np.array(self.freeparamnames) == 'rp'+str(0))[0])
            #logl = -0.5 * ((p[rpind] - 0.05)/0.01)**2
            #return logl

            models = modelobj.makemodel(p)
            logl = [-len(self.photnoiseests[i][self.binnedoks[i]]) * np.log(p[self.sinds[i]]) - 0.5*(1./(p[self.sinds[i]]**2))*np.sum((((self.lcs[i] - models[i])/self.photnoiseests[i])[self.binnedoks[i]])**2) for i in self.rangeofdirectories]
            #for subdir in self.wavebin['subdirectories']:
            #    n = self.inputs[subdir]['n']
                # p[sind] is an 's' parameter; if the uncertainties do not need to be re-scaled then s = 1
                # there is a single 's' parameter for each night's fit - helpful if a dataset is far from the photon noise
            #    sind = np.argwhere(np.array(self.freeparamnames) == 's'+n)[0][0]
            #    penaltyterm = -len(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']]) * np.log(p[sind])
            #    chi2 = (((self.wavebin[subdir]['lc'] - models[subdir])/self.wavebin[subdir]['photnoiseest'])[self.wavebin[subdir]['binnedok']])**2
            #    logl.append(penaltyterm - 0.5*(1./(p[sind]**2))*np.sum(chi2))
            return np.sum(logl)
            
        # inverse transform sampling    
        # calculate inverse cdf (ppf) of gaussian priors on u0 and u1; interpolations can be used to assign values in prior transform
        v = np.linspace(0, 1, 100000)
        ppf_u0 = stats.norm.ppf(v, loc=self.wavebin['ldparams']['v0'], scale=self.wavebin['ldparams']['v0_unc'])
        ppf_func_u0 = interpolate.interp1d(v, ppf_u0)
        ppf_u1 = stats.norm.ppf(v, loc=self.wavebin['ldparams']['v1'], scale=self.wavebin['ldparams']['v1_unc'])
        ppf_func_u1 = interpolate.interp1d(v, ppf_u1)
        u0ind = np.argwhere(np.array(self.freeparamnames) == 'u0'+self.firstn)[0][0]
        u1ind = np.argwhere(np.array(self.freeparamnames) == 'u1'+self.firstn)[0][0]
        def ptform(p):

            x = np.array(p)
            span = self.mcmcbounds[1] - self.mcmcbounds[0]
            x = x*span + self.mcmcbounds[0]

            if x[u0ind] < 0.0001: x[u0ind] = 0.0001     # this prevents trying to interpolate a value that is beyond the bounds of the interpolation
            if x[u0ind] > .9999: x[u0ind] = .9999
            x[u0ind] = ppf_func_u0(x[u0ind])

            if x[u1ind] < 0.0001: x[u1ind] = 0.0001     # this prevents trying to interpolate a value that is beyond the bounds of the interpolation
            if x[u1ind] > .9999: x[u1ind] = .9999
            x[u1ind] = ppf_func_u1(x[u1ind])

            return x

        ndim = len(self.freeparamnames)

        self.speak('running dynesty')
        #self.dsampler = dynhelp.dyn(self.detrender, self.inputs, self.wavebin, self.mcmcbounds, ndim)

        if ndim > 25: # use special inputs that will make run more efficient
            self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, bound='multi', sample='slice', update_interval=float(ndim))
            self.dsampler.run_nested(nlive_init=int(5*ndim), nlive_batch=int(5*ndim), wt_kwargs={'pfrac': 1.0}) # place 100% of the weight on the posterior, don't sample the evidence
        else: # use defaults
            self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, sample='slice')
            self.dsampler.run_nested(wt_kwargs={'pfrac': 1.0})

        quantiles = [_quantile(self.dsampler.results['samples'][:,i], [.16, .5, .84], weights=np.exp(self.dsampler.results['logwt'] - self.dsampler.results['logwt'][-1])) for i in range(len(self.freeparamnames))]
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
        modelobj = ModelMaker(self.inputs, self.wavebin)
        models = modelobj.makemodel(self.mcparams[:,0])
        resid = [(self.lcs[i] - models[i])[self.binnedoks[i]] for i in self.rangeofdirectories]
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('mcfit overall RMS: '+str(data_unc))

        # how many times the expected noise is the rms?
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            self.write('x mean expected noise for {0}: {1}'.format(subdir, np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']])))
        self.write('x median mean expected noise for joint fit: {0}'.format(np.median([np.std(resid[n])/np.mean(self.wavebin[subdir]['photnoiseest'][self.wavebin[subdir]['binnedok']]) for n in range(len(self.wavebin['subdirectories']))])))

        fitmodel = {}
        batmanmodel = {}
        for s, subdir in enumerate(self.wavebin['subdirectories']): 
            fitmodel[subdir] = models.fitmodel[s]
            batmanmodel[subdir] = models.batmanmodel[s]
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

