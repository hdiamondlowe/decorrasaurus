from .imports import *
from .ModelMaker import ModelMaker
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
import corner
from dynesty import plotting as dyplot

class Plotter(Talker):

    '''this class will plot all the things you wish to see'''

    def __init__(self, inputs, subcube):
        ''' initialize the plotter 
        directorypath is optional - it will be used by figures.py after detrender is finished '''
        Talker.__init__(self)

        self.inputs = inputs
        self.subcube = subcube

    def cubeplots(self):
        
        self.speak('plotting raw counts vs wavelength for all stars')
        colors = ['firebrick', 'darkorange', 'olivedrab', 'dodgerblue', 'darkmagenta']
        for n, subdir in enumerate(self.inputs['subdirectories']):      
            target = self.subcube[subdir]['target']
            comparisons = self.subcube[subdir]['comparisons']
            expind = np.where(self.subcube[subdir]['airmass'] == min(self.subcube[subdir]['airmass']))[0][0] # use a low-airmass exposure
            plt.plot(self.subcube[subdir]['wavelengths'], self.subcube[subdir]['raw_counts'][target][expind], alpha=0.75, lw=2, color=colors[n])
            for comp in comparisons:
                plt.plot(self.subcube[subdir]['wavelengths'], self.subcube[subdir]['raw_counts'][comp][expind], alpha=0.75, lw=2, color=colors[n])
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Raw Counts (photoelectrons)')
        plt.tight_layout()
        plt.savefig(self.inputs['directoryname']+'figure_rawspectra.png')
        plt.clf()
        plt.close()
        
    def lcplots(self, wavebin):
        
        self.wavebin = wavebin

        if self.inputs['makeplots']:
            self.speak('plotting binned light curve')
            plt.figure()
            for subdir in self.wavebin['subdirectories']:
                self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])
                target = self.subcube[subdir]['target']
                comparisons = self.subcube[subdir]['comparisons']
                expind = np.argmin(np.array(self.subcube[subdir]['airmass'])) # use the exposer at lowest airmass
                bininds = self.wavebin[subdir]['bininds']

                targcounts = self.subcube[subdir]['raw_counts'][target][expind]*bininds
                where = np.where(targcounts)[0]
                plt.plot(self.subcube[subdir]['wavelengths'], targcounts/np.median(targcounts[where]), alpha=0.75, lw=2, label=target)
                for comp in comparisons:
                    compcounts = self.subcube[subdir]['raw_counts'][comp][expind]*bininds
                    where = np.where(compcounts)
                    plt.plot(self.subcube[subdir]['wavelengths'], compcounts/np.median(compcounts[where]), alpha=0.75, lw=2, label=comp)
                plt.legend(loc='best')
                plt.xlabel(r'Wavelength ($\AA$)', fontsize=20)
                plt.ylabel('Normalized Raw Flux', fontsize=20)
                plt.xlim(self.wavebin['wavelims'][0], self.wavebin['wavelims'][1])
                plt.title(subdir+', '+self.wavefile+' angstroms')
                plt.tight_layout()
                plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_wavebinnedspectrum_'+subdir+'.png')
                plt.clf()
                plt.close()

        self.speak('plotting fit parameters')
        for subdir in self.wavebin['subdirectories']:
            for f in self.inputs[subdir]['fitlabels']:
                plt.plot(self.wavebin[subdir]['compcube']['bjd'][self.wavebin[subdir]['binnedok']], self.wavebin[subdir]['compcube'][f][self.wavebin[subdir]['binnedok']], alpha=0.7, label=f+str(self.inputs[subdir]['n']))
            polymodel = 0
            for p in range(self.inputs[subdir]['polyfit']):
                polymodel += ((self.wavebin[subdir]['compcube']['bjd'] - self.inputs[subdir]['toff'])[self.wavebin[subdir]['binnedok']])**p           
            if self.inputs[subdir]['polyfit'] > 0: plt.plot(self.wavebin[subdir]['compcube']['bjd'][self.wavebin[subdir]['binnedok']], polymodel, label='polyfit deg'+str(self.inputs[subdir]['polyfit']))
            plt.legend(loc='best')
            plt.xlabel('BJD')
            plt.ylabel('Parameter')
            plt.title(subdir+', '+self.wavefile+', decorrelation parameters')
            plt.tight_layout()
            plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_decorrelationparams_'+subdir+'.png')
            plt.clf()
            plt.close()
        
    def lmplots(self, wavebin, linfits):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making lmfit figures')

        self.speak('making model to offset bjd times')
        #lcbinned = self.targcomp_binned.binned_lcs_dict[self.keys[k]]
        modelobj = ModelMaker(self.inputs, self.wavebin, self.wavebin['lmfit']['values'])
        models = modelobj.makemodel()

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C8', 'C9']

        t0 = []
        for subdir in self.wavebin['subdirectories']:
            n = str(self.inputs[subdir]['n'])
            if 'dt'+n in self.wavebin['lmfit']['freeparamnames']:
                dtind = np.argwhere(np.array(self.wavebin['lmfit']['freeparamnames']) == 'dt'+n)[0][0]
                t0.append(self.inputs[subdir]['toff'] + self.wavebin['lmfit']['values'][dtind])
            else:
                dtind = np.argwhere(np.array(self.inputs[subdir]['tranlabels']) == 'dt')[0][0]
                t0.append(self.inputs[subdir]['toff'] + self.inputs[subdir]['tranparams'][dtind])


        if self.inputs['makeplots']:
            for n, subdir in enumerate(self.wavebin['subdirectories']):
                self.speak('plotting normalized wavelength binned raw counts vs time for target and comparisons for {0}'.format(subdir))
                target, comparisons = self.subcube[subdir]['target'], self.subcube[subdir]['comparisons']
                binnedtarg = np.sum(self.subcube[subdir]['raw_counts'][target] * self.wavebin[subdir]['bininds'], 1)
                binnedcomp = np.array([np.sum(self.subcube[subdir]['raw_counts'][comparisons[i]] * self.wavebin[subdir]['bininds'], 1) for i in range(len(comparisons))])
                plt.figure()
                plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (binnedtarg/np.median(binnedtarg))[self.wavebin[subdir]['binnedok']], '.', alpha=0.5, label=target)
                for i,c in enumerate(binnedcomp):
                    plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (c/np.median(c))[self.wavebin[subdir]['binnedok']], '.', alpha=0.5, label=comparisons[i])
                plt.legend(loc='best')
                plt.xlabel('BJD-'+str(t0[n]), fontsize=20)
                plt.ylabel('Normalized Flux', fontsize=20)
                plt.title(subdir+', '+self.wavefile+' angstroms')
                plt.tight_layout()
                plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_wavebinnedtimeseries_'+subdir+'.png')
                plt.clf()
                plt.close()

                self.speak('plotting normalized wavelength binned raw counts vs time for target and summed comparisons for {0}'.format(subdir))
                plt.figure()
                binnedsupercomp = np.sum(binnedcomp, 0)
                plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (binnedtarg/np.median(binnedtarg))[self.wavebin[subdir]['binnedok']], '.', alpha=0.5, label=target)
                plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (binnedsupercomp/np.median(binnedsupercomp))[self.wavebin[subdir]['binnedok']], '.', alpha=0.5, label='summedcomp')
                plt.legend(loc='best')
                plt.xlabel('BJD-'+str(t0[n]), fontsize=20)
                plt.ylabel('Normalized Flux', fontsize=20)
                plt.title(subdir+', '+self.wavefile+' angstroms')
                plt.tight_layout()
                plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_wavebinnedtimeseries_summedcomp_'+subdir+'.png')
                plt.clf()
                plt.close()

        self.speak('making lightcurve and lmfit model vs time figure')
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            plt.figure(figsize=(12, 14))
            gs = plt.matplotlib.gridspec.GridSpec(3, 1, hspace=0.15, wspace=0.15, left=0.08,right=0.98, bottom=0.07, top=0.92)
            lcplots = {}
            lcplots.setdefault('lcplusmodel', []).append(plt.subplot(gs[0:2,0]))
            lcplots.setdefault('residuals', []).append(plt.subplot(gs[2,0]))

            # plot the points that were used in the fit
            lcplots['lcplusmodel'][0].plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], self.wavebin[subdir]['lc'][self.wavebin[subdir]['binnedok']], 'o', color='C'+self.inputs[subdir]['n'], markeredgecolor='none', alpha=0.5)
            # plot the points that were not used
            #lcplots['lcplusmodel'][0].plot(self.subcube[n]['bjd'][np.invert(self.wavebin['binnedok'][n])]-t0[n], self.wavebin['lc'][n][np.invert(self.wavebin['binnedok'][n])], 'ko', markeredgecolor='none', alpha=0.2)
            #lcplots['lcplusmodel'][0].plot(self.subcube[n]['bjd']-t0[n], models[n], 'k-', lw=2, alpha=0.5)
            lcplots['lcplusmodel'][0].plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], models[subdir][self.wavebin[subdir]['binnedok']], 'k-', lw=2, alpha=0.5)
            lcplots['lcplusmodel'][0].set_ylabel('lightcurve + model', fontsize=16)
    
            lcplots['residuals'][0].plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (self.wavebin[subdir]['lc']-models[subdir])[self.wavebin[subdir]['binnedok']], 'o', color='C'+self.inputs[subdir]['n'], markeredgecolor='none', alpha=0.5)
            #lcplots['residuals'][0].plot(self.subcube[n]['bjd'][np.invert(self.wavebin['binnedok'][n])]-t0[n], (self.wavebin['lc'][n]-models[n])[np.invert(self.wavebin['binnedok'][n])], 'ko', markeredgecolor='none', alpha=0.2)
            lcplots['residuals'][0].axhline(0, -1, 1, color='k', linestyle='-', linewidth=2, alpha=0.5)
            lcplots['residuals'][0].set_xlabel('BJD-'+str(t0), fontsize=16)
            lcplots['residuals'][0].set_ylabel('Residuals', fontsize=16)
            plt.suptitle(subdir+', lightcurve plus lmfit model, '+self.wavefile+' angstroms', fontsize=20)
            plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_lmfitlcplusmodel_'+subdir+'.png')
            plt.clf()
            plt.close()

        if self.inputs['dividewhite'] and self.inputs['binlen']!='all':
            self.speak('plotting divide white lmfit detrended lightcurve with batman model vs time')
            plt.figure()
            for n, night in enumerate(self.wavebin['subdirectories']):
                plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (self.wavebin[subdir]['lc']/(modelobj.fitmodel[subdir]*self.wavebin[subdir]['Zwhite']*self.wavebin['Zcomp'][n]))[self.wavebin[subdir]['binnedok']], 'o', markeredgecolor='none', alpha=0.5)
                #plt.plot(self.subcube[n]['bjd'][np.invert(self.wavebin['binnedok'][n])]-t0[n], (self.wavebin['lc'][n]/(modelobj.fitmodel[n]*self.wavebin['Zwhite'][n]*self.wavebin['Zcomp'][n]))[np.invert(self.wavebin['binnedok'][n])], 'ko', markeredgecolor='none', alpha=0.2)
            for n, night in enumerate(self.inputs.subdirectories):
                #plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], modelobj.batmanmodel[n][self.wavebin['binnedok'][n]], 'k-', lw=2, alpha=0.5)
                plt.plot(self.subcube[n]['bjd']-t0[n], modelobj.batmanmodel[n], 'k-', lw=2, alpha=0.5)
            plt.xlabel('time from mid-transit [days]', fontsize=20)
            plt.ylabel('normalized flux', fontsize=20)
            plt.title('lmfit for fit, '+self.wavefile+' angstroms', fontsize=20)
            plt.tight_layout()
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_lmfitdetrendedlc.png')
            plt.clf()
            plt.close()
        else:
            self.speak('plotting lmfit detrended lightcurve with batman model vs time')
            plt.figure()
            for n, subdir in enumerate(self.wavebin['subdirectories']):
                plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (self.wavebin[subdir]['lc']/modelobj.fitmodel[subdir])[self.wavebin[subdir]['binnedok']], 'o', color='C'+self.inputs[subdir]['n'], markeredgecolor='none', alpha=0.5)
                #plt.plot(self.subcube[n]['bjd'][np.invert(self.wavebin['binnedok'][n])]-t0[n], (self.wavebin['lc'][n]/modelobj.fitmodel[n])[np.invert(self.wavebin['binnedok'][n])], 'ko', markeredgecolor='none', alpha=0.2)
            for n, subdir in enumerate(self.wavebin['subdirectories']):
                #plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], modelobj.batmanmodel[n][self.wavebin['binnedok'][n]], 'k-', lw=2, alpha=0.5)
                plt.plot(self.subcube[subdir]['bjd']-t0[n], modelobj.batmanmodel[subdir], 'k-', lw=2, alpha=0.5)
            plt.xlabel('time from mid-transit [days]', fontsize=20)
            plt.ylabel('normalized flux', fontsize=20)
            plt.title('lmfit for fit, '+self.wavefile+' angstroms', fontsize=20)
            plt.tight_layout()
            plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_lmfitdetrendedlc.png')
            plt.clf()
            plt.close()
        
        self.speak('plotting fit residual histogram')
        dist = []
        for subdir in self.wavebin['subdirectories']:
            resid = (self.wavebin[subdir]['lc'] - models[subdir])[self.wavebin[subdir]['binnedok']]
            data_unc = np.std(resid)
            dist.append(resid/data_unc)
        dist = np.hstack(dist)
        n, bins, patches = plt.hist(dist, bins=25, normed=1, color='b', alpha=0.6, label='residuals')
        gaussiandist = np.random.randn(10000)
        ngauss, binsgauss, patchesgauss = plt.hist(gaussiandist, bins=25, normed=1, color='r', alpha=0.6, label='gaussian')
        plt.title('Residuals for '+self.wavefile, fontsize=20)
        plt.xlabel('Uncertainty-Weighted Residuals', fontsize=20)
        plt.ylabel('Number of Data Points', fontsize=20)
        plt.legend()
        plt.savefig(self.inputs['directoryname']+self.wavefile+'_residuals_hist.png')
        plt.clf()
        plt.close()

        self.speak('making lmfit residuals plot for each lmfit call')
        for i, linfit in enumerate(linfits):
            plt.plot(range(len(linfit.residual)), linfit.residual, '.', alpha=.6, label='linfit'+str(i)+', chi2 = '+str(linfit.chisqr))
        plt.xlabel('Data Number')
        plt.ylabel('Residuals')
        plt.legend(loc='best')
        plt.savefig(self.inputs['directoryname']+self.wavefile+'_linfit_residuals.png')
        plt.clf()
        plt.close()
        
        self.speak('making rms vs binsize figure after lmfit')
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            time = (self.subcube[subdir]['bjd'] - t0[n])[self.wavebin[subdir]['binnedok']]     # days
            time = time*24.*60.                       # time now in minutes
            sigma_resid = np.std((self.wavebin[subdir]['lc']-models[subdir])[self.wavebin[subdir]['binnedok']])
            numbins = 1
            bins = []
            rms = []
            gaussianrms = []
            for i in range(len(time)):
                hist = np.histogram(time, numbins)
                ind_bins, time_bins = hist[0], hist[1]      # number of points in each bin (also helps index), bin limits in units of time [days]
                dtime = time_bins[1]-time_bins[0]
                if dtime < (0.5*self.inputs['Tdur']*24.*60.):
                    indlow = 0
                    num = 0
                    for i in ind_bins:
                        num += np.power((np.mean(self.wavebin[subdir]['lc'][self.wavebin[subdir]['binnedok']][indlow:indlow+i] - models[subdir][self.wavebin[subdir]['binnedok']][indlow:indlow+i])), 2.)
                        indlow += i
                    calc_rms = np.sqrt(num/numbins)
                    if np.isfinite(calc_rms) != True: 
                        numbins +=1 
                        continue
                    rms.append(calc_rms)
                    bins.append(dtime)    # bin size in units of days
                    gaussianrms.append(sigma_resid/np.sqrt(np.mean(ind_bins)))
                numbins += 1
            plt.loglog(np.array(bins), gaussianrms, 'r', lw=2, label='std. err.')
            plt.loglog(np.array(bins), rms, 'k', lw=2, label='rms')
            plt.xlim(bins[-1], bins[0])
            plt.xlabel('Bins (min)')
            plt.ylabel('RMS')
            plt.title(r'RMS vs Bin Size for {0}, {1} $\AA$'.format(subdir, self.wavefile))
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_lmfitrmsbinsize'+subdir+'.png')
            plt.clf()
            plt.close()
  
        
        plt.close('all')
    
    def fullplots(self, wavebin):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])
        
        self.speak('making fullfit figures')

        t0 = []
        for subdir in self.wavebin['subdirectories']:
            n = str(self.inputs[subdir]['n'])
            if 'dt'+n in self.wavebin['mcfit']['freeparamnames']:
                dtind = np.argwhere(np.array(self.wavebin['mcfit']['freeparamnames']) == 'dt'+n)[0][0]
                t0.append(self.inputs[subdir]['toff'] + self.wavebin['mcfit']['values'][dtind])
            else:
                dtind = np.argwhere(np.array(self.inputs[subdir]['tranlabels']) == 'dt')[0][0]
                t0.append(self.inputs[subdir]['toff'] + self.inputs[subdir]['tranparams'][dtind])

        modelobj = ModelMaker(self.inputs, self.wavebin, self.wavebin['mcfit']['values'])
        models = modelobj.makemodel()
        self.speak('plotting fullfit detrended lightcurve with batman model vs time')
        plt.figure()
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (self.wavebin[subdir]['lc']/modelobj.fitmodel[subdir])[self.wavebin[subdir]['binnedok']], 'o', color='C'+self.inputs[subdir]['n'], markeredgecolor='none', alpha=0.5)
        for n, subdir in enumerate(self.wavebin['subdirectories']):
            plt.plot(self.subcube[subdir]['bjd'][self.wavebin[subdir]['binnedok']]-t0[n], (modelobj.batmanmodel[subdir])[self.wavebin[subdir]['binnedok']], 'k-', lw=2)
        plt.xlabel('Time from Mid-Transit (days)', fontsize=20)
        plt.ylabel('Normalized Flux', fontsize=20)
        plt.title('Full Fit for '+self.wavefile+' angstroms', fontsize=20)
        plt.tight_layout()
        plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_fullfitdetrendedlc.png')
        plt.clf()
        plt.close()

        '''
        if self.inputs['samplecode'] == 'emcee':        

            self.speak('plotting walkers vs steps')
            fig, axes = plt.subplots(len(self.inputs.freeparamnames), 1, sharex=True, figsize=(16, 12))
            for i, name in enumerate(self.inputs.freeparamnames):
                axes[i].plot(self.wavebin['mcfit']['chain'][:, :, i].T, color="k", alpha=0.4)
                axes[i].yaxis.set_major_locator(MaxNLocator(5))
                axes[i].axhline(self.wavebin['mcfit']['values'][i][0], color="#4682b4", lw=2)
                if self.inputs.freeparambounds[0][i] == True:
                    axes[i].axhline(self.wavebin['mcfit']['values'][i][0]-self.wavebin['lmfit']['uncs'][i], color="#4682b4", lw=2, ls='--')
                else:
                    axes[i].axhline(self.inputs.freeparambounds[0][i], color="#4682b4", lw=2, ls='--')
                if self.inputs.freeparambounds[1][i] == True:
                    axes[i].axhline(self.wavebin['mcfit']['values'][i][0]+self.wavebin['lmfit']['uncs'][i], color="#4682b4", lw=2, ls='--')
                else:
                    axes[i].axhline(self.inputs.freeparambounds[1][i], color="#4682b4", lw=2, ls='--')
                axes[i].axvline(self.inputs.burnin, color='k', ls='--', lw=1, alpha=0.4)
                axes[i].set_ylabel(self.inputs.freeparamnames[i])
                if i == len(self.inputs.freeparamnames)-1: axes[i].set_xlabel("step number")
            fig.subplots_adjust(hspace=0)
            fig.tight_layout()
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_mcmcchains.png')
            plt.clf()
            plt.close()

            
            self.speak('plotting mcmc corner plot')
            samples = self.wavebin['mcfit']['chain'][:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.freeparamnames)))
            fig = corner.corner(samples, labels=self.inputs.freeparamnames, truths=self.wavebin['lmfit']['values'])
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_mcmccorner.png')
            plt.clf()
            plt.close()

        elif self.inputs['samplecode'] == 'dynesty':

            truths = self.wavebin['lmfit']['values'][:]
            truths.append(self.wavebin['ldparams']['v0'])
            truths.append(self.wavebin['ldparams']['v1'])
            for n in range(len(self.wavebin['subdirectories'])): truths.append(1)

            # trace plot
            self.speak('plotting dynesty trace plots')
            fig, axes = dyplot.traceplot(self.wavebin['mcfit']['results'], labels=self.wavebin['mcfit']['freeparamnames'], post_color='royalblue', truths=truths, truth_color='firebrick', truth_kwargs={'alpha': 0.8}, fig=plt.subplots(len(self.wavebin['mcfit']['freeparamnames']), 2, figsize=(12, 30)), trace_kwargs={'edgecolor':'none'})
            plt.savefig(self.inputs['directoryname']+self.wavefile+'_figure_dynestychains.png')
            plt.clf()
            plt.close()

            
            # corner plot
            fig, axes = dyplot.cornerplot(self.wavebin['mcfit']['results'], labels=self.inputs.freeparamnames, truths=truths, show_titles=True, title_kwargs={'y': 1.04}, fig=plt.subplots(len(self.inputs.freeparamnames), len(self.inputs.freeparamnames), figsize=(15, 15)))
            plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmccorner.png')
            plt.clf()
            plt.close()
            '''

