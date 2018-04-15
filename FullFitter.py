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
#import emceehelper as mc

class FullFitter(Talker, Writer):

    '''this class will marginalize over the provided parameters using an mcmc'''

    def __init__(self, detrender, wavefile):
        ''' initialize the lmfitter'''
        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.subcube = self.detrender.cube.subcube
        self.wavefile = wavefile
        self.savewave = self.inputs.saveas+self.wavefils
        
        Writer.__init__(self, self.savewave+'.txt')

        self.wavebin = np.load(self.savewave+'.npy')[()]
        if 'mcfit' in self.wavebin.keys():
            self.speak('mcfit already exists for wavelength bin {0}'.format(self.wavefile))
            if self.inputs.mcmccode == 'emcee':
                if self.wavebin['mcfit']['chain'].shape[1] == self.inputs.nsteps:
                    self.speak('mcfit has completed the number of steps')
                elif self.wavebin['mcfit']['chain'].shape[1] < self.inputs.nsteps:
                    self.speak('extending mcfit for more steps')
                    self.runMCFit_dynesty()
        else: 
            self.speak('running mcfit for wavelength bin {0}'.format(self.wavefile))
            if self.inputs.mcmccode == 'emcee': self.runMCFit_emcee()
            elif self.inputs.mcmccode == 'dynesty': self.runMCFit_dynesty()

    def runMCFit_emcee(self):

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
                residuals.append(self.wavebin['lc'][n] - models[n])
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

        # limb darkening parameters were fixed in the Levenberg-Marquardt fits but now we want to fit for them
        if 'u00' in self.inputs.freeparamnames: pass
        else: 
            # append u0+str(0) to the free paramnames
            self.inputs.freeparamnames.append('u00')
            self.inputs.freeparamvalues.append(self.wavebin['ldparams'][0][0])
            # these additional bounds never acutally get used - Gaussian priors get used in the prior transform function; these are just place holders
            self.inputs.freeparambounds[0].append(0.0)          
            self.inputs.freeparambounds[1].append(1.0)
            # append u1+str(0) to the free paramnames
            self.inputs.freeparamnames.append('u10')
            self.inputs.freeparamvalues.append(self.wavebin['ldparams'][0][1])
            self.inputs.freeparambounds[0].append(0.0)          
            self.inputs.freeparambounds[1].append(1.0)
            # need to re-set the boundary names so that ModelMakerJoint will know to use the same 'u00' and 'u01' for all nights
            for n, night in enumerate(self.inputs.nightname):
                if n == 0: self.inputs.tranbounds[n][0][-2], self.inputs.tranbounds[n][0][-1], self.inputs.tranbounds[n][1][-2], self.inputs.tranbounds[n][1][-1] = True, True, True, True
                else: self.inputs.tranbounds[n][0][-2], self.inputs.tranbounds[n][0][-1], self.inputs.tranbounds[n][1][-2], self.inputs.tranbounds[n][1][-1] = 'Joint', 'Joint', 0.0, 0.0
                
        # add these scaling parameters to fit for; ideally they would be 1 but likely they will turn out slightly higher
        for n, night in enumerate(self.inputs.nightname):
            if 's'+str(n) in self.inputs.freeparamnames: pass
            else:
                self.inputs.freeparamnames.append('s'+str(n))
                self.inputs.freeparamvalues.append(1)
                self.inputs.freeparambounds[0].append(0.01)
                self.inputs.freeparambounds[1].append(10.)

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.inputs.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.inputs.freeparambounds[1]]

        for u in range(len(self.inputs.freeparamnames)):
            if type(self.mcmcbounds[0][u]) == bool and self.mcmcbounds[0][u] == True: self.mcmcbounds[0][u] = self.wavebin['lmfit']['values'][u]-self.wavebin['lmfit']['uncs'][u]*25.
            if type(self.mcmcbounds[1][u]) == bool and self.mcmcbounds[1][u] == True: self.mcmcbounds[1][u] = self.wavebin['lmfit']['values'][u]+self.wavebin['lmfit']['uncs'][u]*25.
        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.inputs.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        # rescaling uncertainties as a free parameter during the fit (Berta, et al. 2011, references therein)
        def lnlike(p):
            # test what the priors look like on their own
            #rpind = int(np.where(np.array(self.inputs.freeparamnames) == 'rp'+str(0))[0])
            #logl = -0.5 * ((p[rpind] - 0.05)/0.01)**2
            #return logl
            modelobj = ModelMakerJoint(self.inputs, self.wavebin, p)
            models = modelobj.makemodel()
            logl = []
            for n, night in enumerate(self.inputs.nightname):
                # p[sind] is an 's' parameter; if the uncertainties do not need to be re-scaled then s = 1
                # there is a single 's' parameter for each night's fit - helpful if a dataset is far from the photon noise
                sind = int(np.where(np.array(self.inputs.freeparamnames) == 's'+str(n))[0])
                penaltyterm = -len(self.wavebin['photnoiseest'][n]) * np.log(p[sind])
                chi2 = ((self.wavebin['lc'][n] - models[n])/self.wavebin['photnoiseest'][n])**2
                logl.append(penaltyterm - 0.5*(1./(p[sind]**2))*np.sum(chi2))
            return np.sum(logl)
            

        # inverse transform sampling    
        # calculate inverse cdf (ppf) of gaussian priors on u0 and u1; interpolations can be used to assign values in prior transform
        v = np.linspace(0, 1, 100000)
        ppf_u0 = stats.norm.ppf(v, loc=self.wavebin['ldparams'][0][0], scale=self.wavebin['ldparams'][1][0])
        ppf_func_u0 = interpolate.interp1d(v, ppf_u0)
        ppf_u1 = stats.norm.ppf(v, loc=self.wavebin['ldparams'][0][1], scale=self.wavebin['ldparams'][1][1])
        ppf_func_u1 = interpolate.interp1d(v, ppf_u1)
        def ptform(p):
            x = np.array(p)
            for i in range(len(x)):
                if self.inputs.freeparamnames[i] == 'u00':
                    if x[i] < 0.0001: x[i] = 0.0001     # this prevents trying to interpolate a value that is beyond the bounds of the interpolation
                    if x[i] > .9999: x[i] = .9999
                    x[i] = ppf_func_u0(x[i])
                elif self.inputs.freeparamnames[i] == 'u10':
                    if x[i] < 0.0001: x[i] = 0.0001
                    if x[i] > .9999: x[i] = .9999
                    x[i] = ppf_func_u1(x[i])
                else: 
                    span = self.mcmcbounds[1][i] - self.mcmcbounds[0][i]
                    x[i] = x[i]*span + self.mcmcbounds[0][i]
            return x

        ndim = len(self.inputs.freeparamnames)

        self.speak('running dynesty')
        #self.dsampler = dynhelp.dyn(self.detrender, self.inputs, self.wavebin, self.mcmcbounds, ndim)

        self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, bound='multi', sample='slice', update_interval=float(ndim))
        self.dsampler.run_nested(nlive_init=int(5*ndim), nlive_batch=int(5*ndim), wt_kwargs={'pfrac': 1.0}) # place 100% of the weight on the posterior, don't sample the evidence

        quantiles = [_quantile(self.dsampler.results['samples'][:,i], [.16, .5, .84], weights=np.exp(self.dsampler.results['logwt'] - self.dsampler.results['logwt'][-1])) for i in range(len(self.inputs.freeparamnames))]
        self.mcparams = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), quantiles))

        self.speak('saving mcfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['mcfit'] = {}
        self.wavebin['mcfit']['results'] = self.dsampler.results
        self.wavebin['mcfit']['values'] = self.mcparams
        np.save(self.savewave, self.wavebin)

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.inputs.freeparamnames[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.inputs.freeparamnames))]

        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                ind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                self.inputs.t0[n] = self.mcparams[ind][0] + self.inputs.toff[n]
                self.speak('mcfit reseting t0 parameter for {0}, transit midpoint = {1}'.format(night, self.inputs.t0[n]))
            self.write('mcfit transit midpoint for {0}: {1}'.format(night, self.inputs.t0[n]))

        #calculate rms from mcfit
        modelobj = ModelMakerJoint(self.inputs, self.wavebin, self.mcparams[:,0])
        models = modelobj.makemodel()
        resid = []
        for n in range(len(self.inputs.nightname)):
            resid.append(self.wavebin['lc'][n] - models[n])
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('mcfit overall RMS: '+str(data_unc))

        # how many times the expected noise is the rms?
        for n, night in enumerate(self.inputs.nightname):
            self.write('x mean expected noise for {0}: {1}'.format(night, np.std(resid[n])/np.mean(self.wavebin['photnoiseest'][n])))
        self.write('x median mean expected noise for joint fit: {0}'.format(np.median([np.std(resid[n])/np.mean(self.wavebin['photnoiseest'][n]) for n in range(len(self.inputs.nightname))])))

        plot = PlotterJoint(self.inputs, self.cube)
        plot.mcplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))

    def limbdarkparams(self, wavestart, waveend, teff=3270, teff_unc=104., 
                            logg=5.06, logg_unc=0.20, z=-0.12, z_unc=0.15):
        self.speak('using ldtk to derive limb darkening parameters')
        filters = BoxcarFilter('a', wavestart, waveend),     # Define passbands - Boxcar filters for transmission spectroscopy
        sc = LDPSetCreator(teff=(teff, teff_unc),             # Define your star, and the code
                           logg=(logg, logg_unc),             # downloads the uncached stellar 
                              z=(z   , z_unc),                # spectra from the Husser et al.
                        filters=filters)                      # FTP server automatically.

        ps = sc.create_profiles()                             # Create the limb darkening profiles
        u , u_unc = ps.coeffs_qd(do_mc=True)                  # Estimate non-linear law coefficients
        self.u0, self.u1 = u[0][0], u[0][1]
        self.u0_unc, self.u1_unc = u_unc[0][0], u_unc[0][1]
        self.write('limb darkening params: '+str(self.u0)+'  '+str(self.u1))

        for n in range(len(self.inputs.nightname)):
            if 'u0' in self.inputs.tranlabels[n]:
                self.inputs.tranparams[n][-2], self.inputs.tranparams[n][-1] = self.u0, self.u1
            if self.inputs.tranbounds[n][0][-2] == 'Joint': continue
            else:
                self.inputs.tranbounds[n][0][-2], self.inputs.tranbounds[n][1][-2] = self.u0-(5.*self.u0_unc), self.u0+(5.*self.u0_unc)
                self.inputs.tranbounds[n][0][-1], self.inputs.tranbounds[n][1][-1] = self.u1-(5.*self.u1_unc), self.u1+(5.*self.u1_unc)
                u0ind = np.where(np.array(self.inputs.freeparamnames) == 'u0'+str(n))[0][0]
                self.inputs.freeparambounds[0][u0ind], self.inputs.freeparambounds[1][u0ind] = self.u0-(5.*self.u0_unc), self.u0+(5.*self.u0_unc)
                self.inputs.freeparambounds[0][u0ind+1], self.inputs.freeparambounds[1][u0ind+1] = self.u1-(5.*self.u1_unc), self.u1+(5.*self.u1_unc)


