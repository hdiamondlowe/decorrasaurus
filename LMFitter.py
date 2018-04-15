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
        self.inputs = self.detrender.inputs
        self.cube = self.detrender.cube
        self.wavefile = wavefile
        
        Writer.__init__(self, self.inputs.saveas+self.wavefile+'.txt')

        self.wavebin = np.load(self.inputs.saveas+self.wavefile+'.npy')[()]
        if 'lmfit' in self.wavebin.keys():
            self.speak('lmfit already exists for wavelength bin {0}'.format(self.wavefile))
        else: 
            self.speak('running lmfit for wavelength bin {0}'.format(self.wavefile))
            self.runLMFit()

    def runLMFit(self):

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        for n, night in enumerate(self.inputs.nightname):
            if 's'+str(n) in self.inputs.freeparamnames: self.inputs.freeparamnames.remove('s'+str(n))

        self.speak('running first lmfit scaling by photon noise limits')#, making output txt file')

        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.inputs.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.inputs.freeparamvalues[n])
            if self.inputs.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.inputs.freeparambounds[0][n]
            if self.inputs.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.inputs.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        def lineareqn(params):
            paramvals = [params[name].value for name in self.inputs.freeparamnames]
            model = ModelMaker(self.inputs, self.wavebin, paramvals)
            return model.makemodel()

        def residuals1(params):
            models = lineareqn(params)
            residuals = []
            for n, night in enumerate(self.inputs.nightname):
                residuals.append((self.wavebin['lc'][n] - models[n])/self.wavebin['photnoiseest'][n]) # weight by photon noise limit (expected noise)
            return np.hstack(residuals)


        fit_kws={'epsfcn':1e-5}  # set the stepsize to something small but reasonable; withough this lmfit may have trouble perturbing values
            #, 'full_output':True, 'xtol':1e-5, 'ftol':1e-5, 'gtol':1e-5}
        self.linfit1 = lmfit.minimize(fcn=residuals1, params=lmfitparams, method='leastsq', **fit_kws)
        linfit1paramvals = [self.linfit1.params[name].value for name in self.inputs.freeparamnames]
        linfit1uncs = np.sqrt(np.diagonal(self.linfit1.covar))
        self.write('1st lm params:')
        [self.write('    '+self.inputs.freeparamnames[i]+'    '+str(linfit1paramvals[i])+'  +/-  '+str(linfit1uncs[i])) for i in range(len(self.inputs.freeparamnames))]

        ######### do a second fit with priors, now that you know what the initial scatter is ########
        
        self.speak('running second lmfit after clipping >{0} sigma points'.format(self.inputs.sigclip))
 
       # median absolute deviation sigma clipping to specified sigma value from inputs
        linfit1paramvals = [self.linfit1.params[name].value for name in self.inputs.freeparamnames]
        modelobj = ModelMaker(self.inputs, self.wavebin, linfit1paramvals)
        models = modelobj.makemodel()
        for n, night in enumerate(self.inputs.nightname):
            resid = self.wavebin['lc'][n] - models[n]
            mad = np.median(abs(resid - np.median(resid)))
            scale = 1.4826
            data_unc = scale*mad               # scale x median absolute deviation
            clip_inds = np.where((resid > (self.inputs.sigclip*data_unc)) | (resid < (-self.inputs.sigclip*data_unc)))[0]
            clip_start = np.where(self.wavebin['binnedok'][n])[0][0]
            self.wavebin['binnedok'][n][clip_start + clip_inds] = False

            # need to update wavebin lc and compcube to reflect data clipping
            newbinnedok = np.ones(self.wavebin['lc'][n].shape, dtype=bool)
            newbinnedok[clip_inds] = False
            self.wavebin['lc'][n] = self.wavebin['lc'][n][newbinnedok]
            self.wavebin['photnoiseest'][n] = self.wavebin['photnoiseest'][n][newbinnedok]
            self.wavebin['compcube'][n] = self.cube.makeCompCube(self.wavebin['bininds'][n], n, self.wavebin['binnedok'][n])
            self.write('clipped points for {0}: {1}'.format(night, clip_start+clip_inds))
            np.save(self.inputs.saveas+self.wavefile, self.wavebin)
            self.speak('remade lc and compcube for {0} in wavebin {1}'.format(night, self.wavefile))


        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.inputs.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.inputs.freeparamvalues[n])
            if self.inputs.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.inputs.freeparambounds[0][n]
            if self.inputs.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.inputs.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)


        def residuals2(params):
            models = lineareqn(params)
            residuals = []
            for n, night in enumerate(self.inputs.nightname):
                residuals.append((self.wavebin['lc'][n] - models[n])/self.wavebin['photnoiseest'][n]) # weight by photon noise limit (expected noise)
            return np.hstack(residuals)

        self.linfit2 = lmfit.minimize(fcn=residuals2, params=lmfitparams, method='leastsq', **fit_kws)
        linfit2paramvals = [self.linfit2.params[name].value for name in self.inputs.freeparamnames]
        linfit2uncs = np.sqrt(np.diagonal(self.linfit2.covar))
        self.write('2nd lm params:')
        [self.write('    '+self.inputs.freeparamnames[i]+'    '+str(linfit2paramvals[i])+'  +/-  '+str(linfit2uncs[i])) for i in range(len(self.inputs.freeparamnames))]

        ######### do a third fit, now with calculated uncertainties ########
        
        self.speak('running third lmfit after calculating undertainties from the data'.format(self.inputs.sigclip))
 
        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.inputs.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.inputs.freeparamvalues[n])
            if self.inputs.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.inputs.freeparambounds[0][n]
            if self.inputs.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.inputs.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        linfit2paramvals = [self.linfit2.params[name].value for name in self.inputs.freeparamnames]
        modelobj = ModelMaker(self.inputs, self.wavebin, linfit2paramvals)
        models = modelobj.makemodel()
        data_uncs2 = []
        for n, night in enumerate(self.inputs.nightname):
            resid = self.wavebin['lc'][n] - models[n]
            data_unc = np.std(resid)
            data_uncs2.append(data_unc)
        self.write('lmfit2 data uncs: {0}'.format(data_uncs2))

        def residuals3(params):
            models = lineareqn(params)
            residuals = []
            for n, night in enumerate(self.inputs.nightname):
                residuals.append((self.wavebin['lc'][n] - models[n])/data_uncs2[n]) # weight by calculated uncertainty
            return np.hstack(residuals)

        self.linfit3 = lmfit.minimize(fcn=residuals3, params=lmfitparams, method='leastsq', **fit_kws)
        linfit3paramvals = [self.linfit3.params[name].value for name in self.inputs.freeparamnames]
        linfit3uncs = np.sqrt(np.diagonal(self.linfit3.covar))
        self.write('3rd lm params:')
        [self.write('    '+self.inputs.freeparamnames[i]+'    '+str(linfit3paramvals[i])+'  +/-  '+str(linfit3uncs[i])) for i in range(len(self.inputs.freeparamnames))]

        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.linfit3.params.keys():
                self.inputs.t0 = self.linfit3.params['dt'+str(n)] + self.inputs.toff
                self.speak('lmfit reseting t0 parameter for {0}, transit midpoint = {1}'.format(night, self.inputs.t0[n]))
            self.write('lmfit transit midpoint for {0}: {1}'.format(night, self.inputs.t0[n]))

        linfit3paramvals = [self.linfit3.params[name].value for name in self.inputs.freeparamnames]
        modelobj = ModelMaker(self.inputs, self.wavebin, linfit3paramvals)
        models = modelobj.makemodel()

        resid = []
        for n in range(len(self.inputs.nightname)):
            resid.append(self.wavebin['lc'][n] - models[n])
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('lmfit overall RMS: '+str(data_unc))  # this is the same as the rms!

        # how many times the expected noise is the rms?
        for n, night in enumerate(self.inputs.nightname):
            self.write('x mean expected noise for {0}: {1}'.format(night, np.std(resid[n])/np.mean(self.wavebin['photnoiseest'][n])))

        # make BIC calculations
        # var = np.power(data_unc, 2.)
        var = 4.5e-7    # variance must remain fixed across all trials in order to make a comparison of BIC values
        for n, night in enumerate(self.inputs.nightname):
            lnlike = -0.5*np.sum((self.wavebin['lc'][n] - models[n])**2/var + np.log(2.*np.pi*var))
            plbls = len(np.where([k.endswith((str(0))) for k in self.linfit3.params.keys()])[0])
            lnn = np.log(len(self.wavebin['lc'][n]))
            BIC = -2.*lnlike + plbls*lnn
            self.write('{0} BIC: {1}'.format(night, BIC))

        self.speak('saving lmfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['lmfit'] = {}
        self.wavebin['lmfit']['values'] = linfit3paramvals
        try: self.wavebin['lmfit']['uncs'] = np.sqrt(np.diagonal(self.linfit3.covar))
        except(ValueError):
            self.speak('the linear fit returned no uncertainties, consider changing tranbounds values')
            return
        if not np.all(np.isfinite(np.array(self.wavebin['lmfit']['uncs']))): 
            self.speak('lmfit error: there were non-finite uncertainties')
            return
        self.wavebin['lmfit']['fitmodels'] = modelobj.fitmodel
        self.wavebin['lmfit']['batmanmodels'] = modelobj.batmanmodel
        np.save(self.inputs.saveas+self.wavefile, self.wavebin)

        plot = Plotter(self.inputs, self.cube.subcube)
        plot.lmplots(self.wavebin, [self.linfit1, self.linfit2, self.linfit3])

        self.speak('done with lmfit for wavelength bin {0}'.format(self.wavefile))

    def limbdarkparams(self, wavestart, waveend):
        self.speak('using ldtk to derive limb darkening parameters')
        filters = BoxcarFilter('a', wavestart, waveend),     # Define passbands - Boxcar filters for transmission spectroscopy
        sc = LDPSetCreator(teff=(self.inputs.Teff, self.inputs.Teff_unc),             # Define your star, and the code
                           logg=(self.inputs.logg, self.inputs.logg_unc),             # downloads the uncached stellar 
                              z=(self.inputs.z   , self.inputs.z_unc),                # spectra from the Husser et al.
                        filters=filters)                      # FTP server automatically.

        ps = sc.create_profiles()                             # Create the limb darkening profiles
        if self.inputs.ldlaw == 'sq': 
            u , u_unc = ps.coeffs_sq(do_mc=True)                  # Estimate non-linear law coefficients
            chains = np.array(ps._samples['sq'])
            u0_array = chains[:,:,0]
            u1_array = chains[:,:,1]
        elif self.inputs.ldlaw == 'qd': 
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
        if self.inputs.ldlaw == 'sq':
            self.v0_array = u0_array/3. + u1_array/5.
            self.v1_array = u0_array/5. - u1_array/3.
        elif self.inputs.ldlaw == 'qd':
            self.v0_array = 2*u0_array + u1_array
            self.v1_array = u0_array - 2*u1_array
        self.v0, self.v1 = np.mean(self.v0_array), np.mean(self.v1_array)
        self.v0_unc, self.v1_unc = np.std(self.v0_array), np.std(self.v1_array)
        self.write('re-parameterized limb darkening params: '+str(self.v0)+' +/- '+str(self.v0_unc)+'    '+str(self.v1)+' +/- '+str(self.v1_unc))

        # save the re-parameterized limb_darkening values so that they can be recalled when making the model
        self.wavebin['ldparams'] = [[self.v0, self.v1], [self.v0_unc, self.v1_unc]]
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)
        self.speak('saved re-parameterized ld values and unertainties')

        for n in range(len(self.inputs.nightname)):
            self.inputs.tranparams[n][-2], self.inputs.tranparams[n][-1] = self.v0, self.v1
            # test uncorrelation of ld params - save original outputs from ldtk
            #self.inputs.tranparams[n][-2], self.inputs.tranparams[n][-1] = u[0][0], u[0][1]
            if self.inputs.tranbounds[n][0][-2] == False: continue
            elif self.inputs.tranbounds[n][0][-2] == 'Joint': continue
            else:
                # put tight 1-sigma bounds on the re-parameterized limb darkening parameters; these will become Gaussian priors in the mcmc or dynesty sampling
                self.inputs.tranbounds[n][0][-2], self.inputs.tranbounds[n][1][-2] = self.v0-self.v0_unc, self.v0+self.v0_unc
                self.inputs.tranbounds[n][0][-1], self.inputs.tranbounds[n][1][-1] = self.v1-self.v1_unc, self.v1+self.v1_unc
                u0ind = np.where(np.array(self.inputs.freeparamnames) == 'u0'+str(n))[0][0]
                self.inputs.freeparambounds[0][u0ind], self.inputs.freeparambounds[1][u0ind] = self.v0-self.v0_unc, self.v0+self.v0_unc
                self.inputs.freeparambounds[0][u0ind+1], self.inputs.freeparambounds[1][u0ind+1] = self.v1-self.v1_unc, self.v1+self.v1_unc


