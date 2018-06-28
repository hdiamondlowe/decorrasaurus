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
        colors = ['firebrick', 'darkorange', 'goldenrod', 'olivedrab', 'dodgerblue', 'darkmagenta']
        for n, subdir in enumerate(self.inputs.subdirectories):      
            target = self.subcube[n]['target']
            comparisons = self.subcube[n]['comparisons']
            expind = np.where(self.subcube[n]['airmass'] == min(self.subcube[n]['airmass']))[0][0] # use a low-airmass exposure
            plt.plot(self.subcube[n]['wavelengths'], self.subcube[n]['raw_counts'][target][expind], alpha=0.75, lw=2, color=colors[n])
            for comp in comparisons:
                plt.plot(self.subcube[n]['wavelengths'], self.subcube[n]['raw_counts'][comp][expind], alpha=0.75, lw=2, color=colors[n])
        plt.xlabel('wavelength (A)')
        plt.ylabel('raw counts (photoelectrons)')
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+'figure_rawspectra.png')
        plt.clf()
        plt.close()

    def lcplots(self, wavebin):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        if self.inputs.makeplots:
            self.speak('plotting binned light curve')
            plt.figure()
            for n, subdir in enumerate(self.inputs.subdirectories):
                target = self.subcube[n]['target']
                comparisons = self.subcube[n]['comparisons']
                expind = np.where(self.subcube[n]['airmass'] == min(self.subcube[n]['airmass']))[0][0] # use the exposer at lowest airmass
                bininds = self.wavebin['bininds'][n]

                targcounts = self.subcube[n]['raw_counts'][target][expind]*bininds
                where = np.where(targcounts)[0]
                plt.plot(self.subcube[n]['wavelengths'], targcounts/np.median(targcounts[where]), alpha=0.75, lw=2, label=target)
                for comp in comparisons:
                    compcounts = self.subcube[n]['raw_counts'][comp][expind]*bininds
                    where = np.where(compcounts)
                    plt.plot(self.subcube[n]['wavelengths'], compcounts/np.median(compcounts[where]), alpha=0.75, lw=2, label=comp)
                plt.legend(loc='best')
                plt.xlabel('wavelength [angstroms]', fontsize=20)
                plt.ylabel('normalized raw flux', fontsize=20)
                plt.xlim(self.wavebin['wavelims'][0], self.wavebin['wavelims'][1])
                plt.title(self.inputs.nightname[n]+', '+self.wavefile+' angstroms')
                plt.tight_layout()
                plt.savefig(self.inputs.saveas+self.wavefile+'_figure_wavebinnedspectrum_'+self.inputs.nightname[n]+'.png')
                plt.clf()
                plt.close()

        self.speak('plotting fit parameters')
        for n, subdir in enumerate(self.inputs.subdirectories):
            for f in self.inputs.fitlabels[n]:
                plt.plot(self.wavebin['compcube'][n]['bjd'][self.wavebin['binnedok'][n]], self.wavebin['compcube'][n][f][self.wavebin['binnedok'][n]], alpha=0.7, label=f+str(n))
            for p in range(self.inputs.polyfit[n]):
                plt.plot(self.wavebin['compcube'][n]['bjd'][self.wavebin['binnedok'][n]], ((self.wavebin['compcube'][n]['bjd'] - self.inputs.toff[n])[self.wavebin['binnedok'][n]])**p, label='polyfit '+str(p))
            plt.legend(loc='best')
            plt.xlabel('bjd')
            plt.ylabel('parameter')
            plt.title(self.inputs.nightname[n]+', '+self.wavefile+', decorrelation parameters')
            plt.tight_layout()
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_decorrelationparams_'+self.inputs.nightname[n]+'.png')
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
        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                dtind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                t0.append(self.inputs.toff[n] + self.wavebin['lmfit']['values'][dtind])
            else:
                dtind = int(np.where(np.array(self.inputs.tranlabels[n]) == 'dt')[0])
                t0.append(self.inputs.toff[n] + self.inputs.tranparams[n][dtind])


        if self.inputs.makeplots:
            for n, subdir in enumerate(self.inputs.subdirectories):
                self.speak('plotting normalized wavelength binned raw counts vs time for target and comparisons for {0}'.format(self.inputs.nightname[n]))
                target, comparisons = self.subcube[n]['target'], self.subcube[n]['comparisons']
                binnedtarg = np.sum(self.subcube[n]['raw_counts'][target] * self.wavebin['bininds'][n], 1)
                binnedcomp = np.array([np.sum(self.subcube[n]['raw_counts'][comparisons[i]] * self.wavebin['bininds'][n], 1) for i in range(len(comparisons))])
                plt.figure()
                plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (binnedtarg/np.median(binnedtarg))[self.wavebin['binnedok'][n]], '.', alpha=0.5, label=target)
                for i,c in enumerate(binnedcomp):
                    plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (c/np.median(c))[self.wavebin['binnedok'][n]], '.', alpha=0.5, label=comparisons[i])
                plt.legend(loc='best')
                plt.xlabel('bjd-'+str(t0[n]), fontsize=20)
                plt.ylabel('normalized flux', fontsize=20)
                plt.title(self.inputs.nightname[n]+', '+self.wavefile+' angstroms')
                plt.tight_layout()
                plt.savefig(self.inputs.saveas+self.wavefile+'_figure_wavebinnedtimeseries_'+self.inputs.nightname[n]+'.png')
                plt.clf()
                plt.close()

                self.speak('plotting normalized wavelength binned raw counts vs time for target and summed comparisons for {0}'.format(self.inputs.nightname[n]))
                plt.figure()
                binnedsupercomp = np.sum(binnedcomp, 0)
                plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (binnedtarg/np.median(binnedtarg))[self.wavebin['binnedok'][n]], '.', alpha=0.5, label=target)
                plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (binnedsupercomp/np.median(binnedsupercomp))[self.wavebin['binnedok'][n]], '.', alpha=0.5, label='summedcomp')
                plt.legend(loc='best')
                plt.xlabel('bjd-'+str(t0[n]), fontsize=20)
                plt.ylabel('normalized nlux', fontsize=20)
                plt.title(self.inputs.nightname[n]+', '+self.wavefile+' angstroms')
                plt.tight_layout()
                plt.savefig(self.inputs.saveas+self.wavefile+'_figure_wavebinnedtimeseries_summedcomp_'+self.inputs.nightname[n]+'.png')
                plt.clf()
                plt.close()

        self.speak('making lightcurve and lmfit model vs time figure')
        plt.figure(figsize=(10, 10))
        for n, subdir in enumerate(self.inputs.subdirectories):
            gs = plt.matplotlib.gridspec.GridSpec(3, 1, hspace=0.15, wspace=0.15, left=0.08,right=0.98, bottom=0.07, top=0.92)
            lcplots = {}
            lcplots.setdefault('lcplusmodel', []).append(plt.subplot(gs[0:2,0]))
            lcplots.setdefault('residuals', []).append(plt.subplot(gs[2,0]))

            # plot the points that were used in the fit
            lcplots['lcplusmodel'][0].plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], self.wavebin['lc'][n][self.wavebin['binnedok'][n]], 'o', markeredgecolor='none', alpha=0.5)
            # plot the points that were not used
            #lcplots['lcplusmodel'][0].plot(self.subcube[n]['bjd'][np.invert(self.wavebin['binnedok'][n])]-t0[n], self.wavebin['lc'][n][np.invert(self.wavebin['binnedok'][n])], 'ko', markeredgecolor='none', alpha=0.2)
            lcplots['lcplusmodel'][0].plot(self.subcube[n]['bjd']-t0[n], models[n], 'k-', lw=2, alpha=0.5)
            lcplots['lcplusmodel'][0].set_ylabel('lightcurve + model', fontsize=20)

            lcplots['residuals'][0].plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (self.wavebin['lc'][n]-models[n])[self.wavebin['binnedok'][n]], 'o', markeredgecolor='none', alpha=0.5)
            #lcplots['residuals'][0].plot(self.subcube[n]['bjd'][np.invert(self.wavebin['binnedok'][n])]-t0[n], (self.wavebin['lc'][n]-models[n])[np.invert(self.wavebin['binnedok'][n])], 'ko', markeredgecolor='none', alpha=0.2)
            lcplots['residuals'][0].axhline(0, -1, 1, color='k', linestyle='-', linewidth=2, alpha=0.5)
            lcplots['residuals'][0].set_xlabel('bjd-'+str(t0), fontsize=20)
            lcplots['residuals'][0].set_ylabel('residuals', fontsize=20)
            plt.suptitle(self.inputs.nightname[n]+'lightcurve plus lmfit model, '+self.wavefile+' angstroms')
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_lmfitlcplusmodel_'+self.inputs.nightname[n]+'.png')
            plt.clf()
            plt.close()

        self.speak('plotting lmfit detrended lightcurve with batman model vs time')
        plt.figure()
        for n, night in enumerate(self.inputs.subdirectories):
            plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (self.wavebin['lc'][n]/modelobj.fitmodel[n])[self.wavebin['binnedok'][n]], 'o', markeredgecolor='none', alpha=0.5)
            #plt.plot(self.subcube[n]['bjd'][np.invert(self.wavebin['binnedok'][n])]-t0[n], (self.wavebin['lc'][n]/modelobj.fitmodel[n])[np.invert(self.wavebin['binnedok'][n])], 'ko', markeredgecolor='none', alpha=0.2)
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

        
        self.speak('plotting fit residual histogram')
        dist = []
        for n, night in enumerate(self.inputs.subdirectories):
            resid = (self.wavebin['lc'][n] - models[n])[self.wavebin['binnedok'][n]]
            data_unc = np.std(resid)
            dist.append((self.wavebin['lc'][n] - models[n])[self.wavebin['binnedok'][n]]/data_unc)
        dist = np.hstack(dist)
        n, bins, patches = plt.hist(dist, bins=25, normed=1, color='b', alpha=0.6, label='residuals')
        gaussiandist = np.random.randn(10000)
        ngauss, binsgauss, patchesgauss = plt.hist(gaussiandist, bins=25, normed=1, color='r', alpha=0.6, label='gaussian')
        plt.title('residuals for '+self.wavefile, fontsize=20)
        plt.xlabel('uncertainty-weighted residuals', fontsize=20)
        plt.ylabel('number of data points', fontsize=20)
        plt.legend()
        plt.savefig(self.inputs.saveas+self.wavefile+'_residuals_hist.png')
        plt.clf()
        plt.close()

        self.speak('making lmfit residuals plot for each lmfit call')
        for i, linfit in enumerate(linfits):
            plt.plot(range(len(linfit.residual)), linfit.residual, '.', alpha=.6, label='linfit'+str(i)+', chi2 = '+str(linfit.chisqr))
        plt.xlabel('data number')
        plt.ylabel('residuals')
        plt.legend(loc='best')
        plt.savefig(self.inputs.saveas+self.wavefile+'_linfit_residuals.png')
        plt.clf()
        plt.close()
        

        plt.close('all')

    def fullplots(self, wavebin):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making fullfit figures')

        if self.inputs.samplecode == 'emcee':        

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

        elif self.inputs.samplecode == 'dynesty':

            truths = self.wavebin['lmfit']['values'][:]
            u0ind = int(np.where(np.array(self.inputs.freeparamnames) == 'u00')[0])
            truths.append(self.inputs.freeparamvalues[u0ind])
            truths.append(self.inputs.freeparamvalues[u0ind+1])
            for n in range(len(self.inputs.nightname)): truths.append(1)

            # trace plot
            self.speak('plotting dynesty trace plots')
            fig, axes = dyplot.traceplot(self.wavebin['mcfit']['results'], labels=self.inputs.freeparamnames, post_color='royalblue', truths=truths, truth_color='firebrick', truth_kwargs={'alpha': 0.8}, fig=plt.subplots(len(self.inputs.freeparamnames), 2, figsize=(12, 30)), trace_kwargs={'edgecolor':'none'})
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_mcmcchains.png')
            plt.clf()
            plt.close()

            '''
            # corner plot
            fig, axes = dyplot.cornerplot(self.wavebin['mcfit']['results'], labels=self.inputs.freeparamnames, truths=truths, show_titles=True, title_kwargs={'y': 1.04}, fig=plt.subplots(len(self.inputs.freeparamnames), len(self.inputs.freeparamnames), figsize=(15, 15)))
            plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmccorner.png')
            plt.clf()
            plt.close()
            '''

        t0 = []
        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                dtind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                t0.append(self.inputs.toff[n] + self.wavebin['mcfit']['values'][dtind][0])
            else:
                dtind = int(np.where(np.array(self.inputs.tranlabels[n]) == 'dt')[0])
                t0.append(self.inputs.toff[n] + self.inputs.tranparams[n][dtind])

        modelobj = ModelMaker(self.inputs, self.wavebin, self.wavebin['mcfit']['values'][:,0])
        models = modelobj.makemodel()
        self.speak('plotting fullfit detrended lightcurve with batman model vs time')
        plt.figure()
        for n, night in enumerate(self.inputs.nightname):
            plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (self.wavebin['lc'][n]/modelobj.fitmodel[n])[self.wavebin['binnedok'][n]], 'o', markeredgecolor='none', alpha=0.5)
        for n, night in enumerate(self.inputs.nightname):
            plt.plot(self.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], (modelobj.batmanmodel[n])[self.wavebin['binnedok'][n]], 'k-', lw=2)
        plt.xlabel('time from mid-transit [days]', fontsize=20)
        plt.ylabel('normalized flux', fontsize=20)
        plt.title('fullfit for fit, '+self.wavefile+' angstroms', fontsize=20)
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+self.wavefile+'_figure_fullfitdetrendedlc.png')
        plt.clf()
        plt.close()

