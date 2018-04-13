from .imports import *
from .ModelMaker import ModelMaker
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
import corner
from dynesty import plotting as dyplot

class Plotter(Talker):

    '''this class will plot all the things you wish to see'''

    def __init__(self, inputs, cube):
        ''' initialize the plotter 
        directorypath is optional - it will be used by figures.py after detrender is finished '''
        Talker.__init__(self)

        self.inputs = inputs
        self.cube = cube

    def lmplots(self, wavebin, linfits):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making lmfit figures')

        self.speak('making model to offset bjd times')
        #lcbinned = self.targcomp_binned.binned_lcs_dict[self.keys[k]]
        modelobj = ModelMaker(self.inputs, self.wavebin, self.wavebin['lmfit']['values'])
        models = modelobj.makemodel()

        t0 = []
        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                dtind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                t0.append(self.inputs.toff[n] + self.wavebin['lmfit']['values'][dtind])
            else:
                dtind = int(np.where(np.array(self.inputs.tranlabels[n]) == 'dt')[0])
                t0.append(self.inputs.toff[n] + self.inputs.tranparams[n][dtind])

        self.speak('making lmfit detrended lightcurve with batman model vs time figure')
        plt.figure()
        for n, night in enumerate(self.inputs.nightname):
            plt.plot(self.cube.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], self.wavebin['lc'][n]/modelobj.fitmodel[n], 'o', alpha=0.5)
        for n, night in enumerate(self.inputs.nightname):
            plt.plot(self.cube.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], modelobj.batmanmodel[n], 'k-', lw=2)
        plt.xlabel('time from mid-transit [days]', fontsize=20)
        plt.ylabel('normalized flux', fontsize=20)
        plt.title('lmfit for fit, '+self.wavefile+' angstroms', fontsize=20)
        #plt.tight_layout()
        plt.savefig(self.inputs.saveas+self.wavefile+'_figure_lmfitdetrendedlc.png')
        plt.clf()
        plt.close()

        
        self.speak('making fit residual histogram figure')
        dist = []
        for n, night in enumerate(self.inputs.nightname):
            resid = self.wavebin['lc'][n] - models[n]
            data_unc = np.std(resid)
            dist.append((self.wavebin['lc'][n] - models[n])/data_unc)
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

    def mcplots(self, wavebin):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making mcfit figures')

        if self.inputs.mcmccode == 'emcee':        

            self.speak('making walkers vs steps figure')
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
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_mcmcchains.png')
            plt.clf()
            plt.close()

            
            self.speak('making mcmc corner plot')
            samples = self.wavebin['mcfit']['chain'][:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.freeparamnames)))
            fig = corner.corner(samples, labels=self.inputs.freeparamnames, truths=self.wavebin['lmfit']['values'])
            plt.savefig(self.inputs.saveas+self.wavefile+'_figure_mcmccorner.png')
            plt.clf()
            plt.close()

        elif self.inputs.mcmccode == 'dynesty':

            truths = self.wavebin['lmfit']['values']
            for n in range(len(self.inputs.nightname)): truths.append(1)

            # trace plot
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
        self.speak('making mcfit detrended lightcurve with batman model vs time figure')
        plt.figure()
        for n, night in enumerate(self.inputs.nightname):
            plt.plot(self.cube.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], self.wavebin['lc'][n]/modelobj.fitmodel[n], 'o', alpha=0.5)
        for n, night in enumerate(self.inputs.nightname):
            plt.plot(self.cube.subcube[n]['bjd'][self.wavebin['binnedok'][n]]-t0[n], modelobj.batmanmodel[n], 'k-', lw=2)
        plt.xlabel('time from mid-transit [days]', fontsize=20)
        plt.ylabel('normalized flux', fontsize=20)
        plt.title('mcfit for fit, '+self.wavefile+' angstroms', fontsize=20)
        #plt.tight_layout()
        plt.savefig(self.inputs.saveas+self.wavefile+'_figure_mcfitdetrendedlc.png')
        plt.clf()
        plt.close()


