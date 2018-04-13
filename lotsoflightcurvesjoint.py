import sys
sys.path.append('/home/hdiamond/local/lib/python2.7/site-packages/')
sys.path.append('/h/mulan0/code/')
sys.path.append('/h/mulan0/code/mosasaurus')
sys.path.append('/h/mulan0/code/detrendersaurus')
import numpy as np
import matplotlib.pyplot as plt
import astrotools.modeldepths as md
import astropy.units as u
from ModelMakerJoint import ModelMakerJoint
from InputsJoint import InputsJoint
import collections
import os
from datetime import datetime
from ldtk import LDPSetCreator, BoxcarFilter
import pickle
from scipy.signal import resample
from scipy.special import gammainc

# this code will take the path to a folder that includes directories of outputs form the detrender
# the output will be a figure of the lightcurves and the transmission spectrum (if flagged)
# if there are many lightcurves in one folder, these will be stacked vertically
# if there are multiple nights, they will be stacked horizontally
# you may need to change the basepath variable to suit your directory setup

# where the analysis files are stored
basepath = '/h/mulan0/analysis/GJ1132/alldata/'
directoryname = 'joint'
#rundir = '2017-09-25-18:37_bin200A-dynesty/'
#rundir = 'bin200A-dynesty-compilation/'
#rundir = 'bin200A-dynestynew-compilation/'
rundir = 'bin200A-dynestynew-s-compilation/'
#rundir = '2017-11-21-16:42_bin200A-dynesty/'

# get model transmission spectrum
specpath = '/home/hdiamond/GJ1132/Spectra/'
specpickles = ['GJ1132b_H_2_350K_scaled.p', 'GJ1132b_H_2_400K_scaled.p', 'GJ1132b_H_2_450K_scaled.p', 'GJ1132b_H_2_500K_scaled.p', 'GJ1132b_H_2_550K_scaled.p', 'GJ1132b_H_2_600K_scaled.p']
model_dicts = [pickle.load(open(specpath+p, 'rb')) for p in specpickles]
model_atmo_temps = [350, 400, 450, 500, 550, 600]
species = 'H_2'

#transmission constants
bin_size = 0.0200          # microns
wave_range = [.7, 1.03]   # microns
smooth = [51, 5]
scale = 1.0
otherpplsdata = False
colors = ['royalblue', 'firebrick', 'goldenrod', 'forestgreen', 'purple']
nightcolor = ['tomato', 'orange', 'greenyellow', 'aqua', 'fuchsia']

#H2 = md.model_spectrum(specdatafile1, bin_size, wave_range, smooth)
#modelspectra = [H2]

# for weighting of model spectra you will need an M dwarf spectrum - trying to average the specrrum across night...
#utnightname = ['ut160227_28', 'ut160303_04', 'ut160308_09', 'ut160416_17', 'ut160421_22']
#spectra = [np.load('/h/mulan0/data/working/GJ1132b_'+utnightname[i]+'/multipleapertures/'+inputs.target[i]+'/extracted'+inputs.mastern[i]+'.npy')[()] for i in range(len(utnightname))]

subdirectories = os.listdir(basepath+directoryname)
loldict = {}

def limbdarkparams(wavestart, waveend, teff=3270, teff_unc=104., 
                        logg=5.06, logg_unc=0.20, z=-0.12, z_unc=0.15):
    print ('        using ldtk to derive limb darkening parameters')
    filters = BoxcarFilter('a', wavestart, waveend),     # Define passbands - Boxcar filters for transmission spectroscopy
    sc = LDPSetCreator(teff=(teff, teff_unc),             # Define your star, and the code
                       logg=(logg, logg_unc),             # downloads the uncached stellar 
                          z=(z   , z_unc),                # spectra from the Husser et al.
                    filters=filters)                      # FTP server automatically.

    ps = sc.create_profiles()                             # Create the limb darkening profiles
    u , u_unc = ps.coeffs_qd(do_mc=True)                  # Estimate non-linear law coefficients
    u0, u1 = u[0][0], u[0][1]
    u0unc, u1unc = u_unc[0][0], u_unc[0][1]
    print '     limb darkening params: '+str(u0)+'  '+str(u1)

    for n in range(len(inputs.nightname)):
        if 'u0' in inputs.tranlabels[n]:
            inputs.tranparams[n][-2], inputs.tranparams[n][-1] = u0, u1
        else:
            # !Error! you also have to add to tranparambounds!
            inputs.tranlabels[n].append('u0')
            inputs.tranparams[n].append(u0)
            inputs.tranlabels[n].append('u1')
            inputs.tranparams[n].append(u1)
    return [[u0, u0unc], [u1, u1unc]]

def loadrun(directory, inputs, wavefile):
    wavebin = np.load(directory+'joint_'+wavefile+'.npy')[()]

    rprsind = int(np.where(np.array(inputs.freeparamnames) == 'rp0')[0])
    if inputs.ldmodel:
        ldparams = limbdarkparams(wavebin['wavelims'][0]/10., wavebin['wavelims'][1]/10.)
    try:
        rprs = wavebin['mcfit']['values'][rprsind][0]
        rprsunc = np.array([wavebin['mcfit']['values'][rprsind][1], wavebin['mcfit']['values'][rprsind][2]])
        modelobj = ModelMakerJoint(inputs, wavebin, wavebin['mcfit']['values'][:,0])
        models = modelobj.makemodel()
        print '        using mcfit values'
    except(KeyError):
        rprs = wavebin['lmfit']['values'][rprsind]
        rprsunc = wavebin['lmfit']['uncs'][rprsind]
        modelobj = ModelMakerJoint(inputs, wavebin, wavebin['lmfit']['values'])
        models = modelobj.makemodel()
        print '        using lmfit values'
    depth = rprs**2
    depthunc = depth*2*(rprsunc/rprs)
    print '        depth [%]', depth*100.
    print '        depth unc [%, %]', depthunc*100.

    resid = []
    for n, night in enumerate(inputs.nightname):
        loldict[night][wavefile]['lc'] = wavebin['lc'][n]
        loldict[night][wavefile]['binnedok'] = wavebin['binnedok'][n]
        loldict[night][wavefile]['fitmodel'] = modelobj.fitmodel[n]
        loldict[night][wavefile]['batmanmodel'] = modelobj.batmanmodel[n]
        resid.append(wavebin['lc'][n] - models[n])
        print '        x expected noise for {0}: {1}'.format(night, np.std(resid[n])/np.median(wavebin['photnoiseest'][n]))
    print '        median expected noise for joint fit: {0}'.format(np.median([np.std(resid[n])/np.median(wavebin['photnoiseest'][n]) for n in range(len(inputs.nightname))]))
    loldict['joint'][wavefile]['depth'] = depth
    loldict['joint'][wavefile]['depthunc'] = depthunc


    for n, night in enumerate(inputs.nightname):
        for f, flabel in enumerate(inputs.fitlabels[n]):
            find = int(np.where(np.array(inputs.freeparamnames) == flabel+str(n))[0])
            try: 
                fvalue = wavebin['mcfit']['values'][find][0]
                func = np.array(wavebin['mcfit']['values'][find][1], wavebin['mcfit']['values'][find][2])
            except(KeyError): 
                fvalue = wavebin['lmfit']['values'][find]
                func = np.array(wavebin['lmfit']['uncs'][find])
            loldict['joint']['parameters'][night][flabel].append(fvalue)
            loldict['joint']['parameters'][night][flabel+'unc'].append(func)

    loldict['joint']['parameters']['u0'].append(ldparams[0][0])
    loldict['joint']['parameters']['u0unc'].append(ldparams[0][1])
    loldict['joint']['parameters']['u1'].append(ldparams[1][0])
    loldict['joint']['parameters']['u1unc'].append(ldparams[1][1])

path = basepath+directoryname+'/run/'+rundir
jointdirectories = [d for d in os.listdir(basepath+directoryname) if os.path.isdir(os.path.join(basepath+directoryname, d))]
if 'run' in jointdirectories: jointdirectories.remove('run')
jointdirectories = sorted(jointdirectories, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))

print 'reading inputs from ', basepath+directoryname
inputs = InputsJoint(jointdirectories, path)
print 'loading joint cube'
subcube = np.load(path+'joint_subcube.npy')[()]
for n, night in enumerate(inputs.nightname):
    loldict[night] = collections.defaultdict(dict)
    loldict[night]['bjd'] = subcube[n]['bjd']
    loldict[night]['t0'] = inputs.t0[n]     # make sure this number is correct - may have to change in input files
loldict['joint'] = collections.defaultdict(dict)
loldict['joint']['wavelims'] = inputs.wavelength_lims
numbins = np.floor((inputs.wavelength_lims[1] - inputs.wavelength_lims[0])/inputs.binlen)
binlen = (inputs.wavelength_lims[1] - inputs.wavelength_lims[0])/numbins
loldict['joint']['binlen'] = binlen

print 'reading wavefiles...'
wavefiles = subcube[0]['wavebin']['wavefiles']
loldict['joint']['wavefiles'] = wavefiles
loldict['joint']['parameters'] = {}

for n, night in enumerate(inputs.nightname):
    loldict['joint']['parameters'][night] = {}
    for f, flabel in enumerate(inputs.fitlabels[0]):
        loldict['joint']['parameters'][night][flabel] = []
        loldict['joint']['parameters'][night][flabel+'unc'] = []
    loldict['joint']['parameters'][night]['flux'] = []
loldict['joint']['parameters']['u0'] = []
loldict['joint']['parameters']['u0unc'] = []
loldict['joint']['parameters']['u1'] = []
loldict['joint']['parameters']['u1unc'] = []

for w in wavefiles:
    print '    wavelength bin', w
    loadrun(path, inputs, w)


##################################################################################
############################ plot lightcurves ####################################
##################################################################################
def lightcurves():
    plt.figure(figsize=(20, 12))
    gs = plt.matplotlib.gridspec.GridSpec(16, 2, hspace=0.0, wspace=0.12, left=0.05,right=0.98, bottom=0.05, top=0.98)
    lcplots = {}
    offset = 0.01

    wavefiles = loldict['joint']['wavefiles']
    #lcplots[0] = {}
    #lcplots[0][0] = plt.subplot(gs[0:, 0])
    #lcplots.setdefault(0, []).append(plt.subplot(gs[:,0]))
    for w in range(len(wavefiles)):
        lcplots[w] = {}
        lcplots[w][1] = plt.subplot(gs[w,1])
        #lcplots.setdefault(w, []).append(plt.subplot(gs[w,1]))
        lcplots[w][0] = plt.subplot(gs[w,0])
    #else: lcplots.setdefault(i, []).append(plt.subplot(gs[0,i], sharey=lcplots[0][0]))


    for w, wave in enumerate(wavefiles):
        resid = []
        for n, night in enumerate(inputs.nightname):
            binnedok = loldict[night][wave]['binnedok']
            t0 = loldict[night]['t0']
            bjdbinned = (loldict[night]['bjd'][binnedok] - t0)
            resid.append(loldict[night][wave]['lc'] - loldict[night][wave]['fitmodel']*loldict[night][wave]['batmanmodel'])

            # need to deal with self.t0 changing during run; this is a secondary issue as you will likely only be plotting from runs where you have fixed self.t0

            lcplots[w][0].plot(bjdbinned, (loldict[night][wave]['lc']/loldict[night][wave]['fitmodel']), 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.5)
            lcplots[w][0].plot(bjdbinned, (loldict[night][wave]['batmanmodel']), color='k', lw=2, alpha=0.8)
            lcplots[w][0].set_xlim(-.05, .05)
            lcplots[w][0].set_ylim(.993, 1.007)
            if w == 8: lcplots[w][0].set_yticks([.993, 1.0, 1.007])
            else: lcplots[w][0].set_yticks([])


            lcplots[w][1].plot(bjdbinned, (loldict[night][wave]['lc'] - (loldict[night][wave]['fitmodel']*loldict[night][wave]['batmanmodel']))*1e6, 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.5)
            lcplots[w][1].axhline(0, 0.025, 1-.025, color='k', lw=2, alpha=0.8)
            lcplots[w][1].set_xlim(-.05, .05)
            lcplots[w][1].set_ylim(-5000, 5000)
            if w == 8: lcplots[w][1].set_yticks([-5000, 0, 5000])
            else: lcplots[w][1].set_yticks([])
            #if w == len(wavefiles)-1: pass
            #else: lcplots[0][1].axhline(1-offset*w-0.5*offset, 0, 1, color='k', lw=1)

            #lcplots[0][0].set_xlim(-0.05, 0.05)
        allresid = np.hstack(resid)
        rms = np.std(allresid)
        lcplots[w][0].text(-0.048, 1.003, wave+' A', color='k', fontsize=14)   #'#ff471a'
        #lcplots[w][1].text(-0.048, 2250, 'RMS: {0:.0f} ppm'.format(rms*1e6), color='k', fontsize=14)
        #lcplots[0][0].text(0.0125, 1.0025-offset*k, 'x exp. noise: {0:.2f}'.format(mcmcRMS/expnoise), color='#ff471a', weight='bold', fontsize=12)

    lcplots[15][0].set_xlabel('time from mid-transit [min]', fontsize=15)
    lcplots[15][0].tick_params(axis='x', labelsize=12)

    lcplots[8][0].set_ylabel('normalized flux', fontsize=15)
    lcplots[8][0].tick_params(axis='y', labelsize=12)
    #plt.ylim(0.9955, 1.002)
    #lcplots[8][0].set_ylim(.84, 1.006)
    #plt.suptitle('wavelength range: ' + str(loldict['joint']['wavelims']) + ' A, binsize: ' + str(loldict['joint']['binlen']) + ' A', fontsize=15)
    lcplots[15][1].set_xlabel('time from mid-transit [min]', fontsize=15)
    lcplots[15][1].tick_params(axis='x', labelsize=12)

    lcplots[8][1].set_ylabel('residuals (ppm)', fontsize=15)
    lcplots[0][1].tick_params(axis='y', labelsize=12)
    #lcplots[0][1].set_yticks([.915, .92, .925])#np.array([.995, 1., 1.005]), ('-2500', '0', '2500'))
    #lcplots[0][1].set_yticklabels(['-2500', '0', '2500'])
    #lcplots[0][1].set_yticks(np.array([.995, 1., 1.005]), ('-2500', '0', '2500'))
    #plt.ylim(0.9955, 1.002)
    #lcplots[0][1].set_ylim(.84, 1.006)
    plt.show()


def transmission(flag='absolute', model=True):
    plt.figure(figsize=(20, 8))

    if flag == 'absolute':

        if model:
            #for m in model_dicts:
                #model_wavelengths, model_depths, model_binned_wavelengths, model_binned_depths = np.array(m[0]), np.array(m[1]), np.array(m[2]), np.array(m[3])
                #plt.plot(model_wavelengths, model_depths*scale*100., color='k', alpha=0.5, linewidth=2, label=r'$100\%\ H_2/He$')
                #plt.plot(model_wavelengths, [np.mean(model_depths*scale*100.) for i in model_wavelengths], color='k', ls='--', alpha=0.5, linewidth=2, label=r'$\mathrm{flat}$')
                #plt.plot(model_binned_wavelengths, model_binned_depths*scale*100., 'ks', alpha=0.75)

            wavefiles = loldict['joint']['wavefiles']
            wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

            xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
            depth =  [loldict['joint'][w]['depth']*100. for w in wavefiles]
            depthunc =  [np.mean(loldict['joint'][w]['depthunc']*100.) for w in wavefiles]

            # linear fit to the transit depths
            z = np.polyfit(xaxis, depth, deg=1)
            p = np.poly1d(z)
            x = np.linspace(wave_range[0], wave_range[1], 100)

            fit_binned_depths = []
            for i in range(len(xaxis)):
                bininds = np.where((x >= wavefilelims[i][0]/10000.) & (x <= wavefilelims[i][1]/10000.))[0]
                fit_binned_depths.append(np.mean(p(x[bininds])))
            fit_binned_depths = np.array(fit_binned_depths)

            Mdwarfspec = subcube[0]['raw_counts'][inputs.target[0]][inputs.targetpx[0]][200]#/np.median(subcube[0]['raw_counts'][inputs.target[0]][inputs.targetpx[0]][200])
            Mdwarfwave = subcube[0]['wavelengths'][inputs.target[0]][inputs.targetpx[0]][200]

            for n, m in enumerate(model_dicts):
                model_wavelengths, model_depths = m[species]['wavelengths'].value, m[species]['model_scaled'].value
                model_wavelengths_binned = []
                model_binned_depths = []
                model_binned_depths_weighted = []
                for l in range(len(xaxis)):
                    bininds = np.where((model_wavelengths >= wavefilelims[l][0]/10000.) & (model_wavelengths <= wavefilelims[l][1]/10000.))[0]
                    Mdwarfbininds = np.where((Mdwarfwave >= wavefilelims[l][0]) & (Mdwarfwave <= wavefilelims[l][1]))[0]
                    newMdwarfspec = resample(Mdwarfspec[Mdwarfbininds], len(bininds))
                    model_wavelengths_binned.append(np.mean(model_wavelengths[bininds]))
                    model_binned_depths.append(np.mean(model_depths[bininds]*scale*100.))
                    model_binned_depths_weighted.append((np.sum(model_depths[bininds]*newMdwarfspec)/np.sum(newMdwarfspec))*scale*100.)
                model_binned_depths = np.array(model_binned_depths)
                model_binned_depths_weighted = np.array(model_binned_depths_weighted)

                # figure out how to move the atmosphere model to best match the points (only shifting, not chaning the size of the features)
                chisq_H2 = np.inf
                best_offset = 0.0
                chisq_H2_weighted = np.inf
                best_offset_weighted = 0.0
                for i in np.linspace(-.1, .1, 1000):
                    model_offset = model_binned_depths + i
                    model_offset_weighted = model_binned_depths_weighted + i
                    chisq_offset = np.sum(((depth - model_offset)/np.mean(depthunc))**2)
                    chisq_offset_weighted = np.sum(((depth - model_offset_weighted)/np.mean(depthunc))**2)
                    if chisq_offset < chisq_H2: 
                        chisq_H2 = chisq_offset
                        best_offset = i
                    if chisq_offset_weighted < chisq_H2_weighted:
                        chisq_H2_weighted = chisq_offset_weighted
                        best_offset_weighted = i
            
                model_binned_depths = model_binned_depths + best_offset
                model_binned_depths_weighted = model_binned_depths_weighted + best_offset_weighted

                print 'best offset:', best_offset
                chisq = np.sum(np.power((depth - model_binned_depths)/np.mean(depthunc), 2.))
                print 'chisq_H2:', chisq

                print 'best offset weighted:', best_offset_weighted
                chisq_weighted = np.sum(np.power((depth - model_binned_depths_weighted)/np.mean(depthunc), 2.))
                print 'chisq_H2 weighted:', chisq_weighted
                pval_weighted = 1 - gammainc(0.5*(len(wavefiles)-1), 0.5*chisq_weighted)
                print 'weighted p-value:', pval_weighted

                plt.plot(model_wavelengths, model_depths*scale*100.+best_offset, color='k', alpha=0.5, linewidth=2, label=r'Solar composition, {0}K, pvalue = {1}'.format(model_atmo_temps[n], '%.3f'%pval_weighted))
                #plt.plot(model_wavelengths_binned, model_binned_depths,'ks', markersize=8, alpha=0.6)
                plt.plot(model_wavelengths_binned, model_binned_depths_weighted,'ks', markersize=8, alpha=0.6)

            chisq_flat = np.sum(np.power((depth - np.array([np.median(depth) for i in depth]))/np.mean(depthunc), 2.))
            print 'chisq_flat:', chisq_flat
            pval_flat = 1 - gammainc(0.5*(len(wavefiles)-1), 0.5*chisq_flat)
            print 'pval flat:', 
            chisq_fit = np.sum(np.power((depth - fit_binned_depths)/np.mean(depthunc), 2.))
            print 'chisq_fit:', chisq_fit
            pval_fit = 1 - gammainc(0.5*(len(wavefiles)-1), 0.5*chisq_fit)

            plt.plot(x, [np.median(depth) for i in x], 'k--', lw=2, alpha=0.8, label=r'flat fit, pvalue = {0}'.format('%.3f'%pval_flat))
            plt.plot(x, p(x), 'k:', lw=3, alpha=0.6, label=r'linear fit, pvalue = {0}'.format('%.3f'%pval_fit))

            plt.errorbar(xaxis, depth, yerr=depthunc, fmt='o', markersize=10, color=colors[0], markeredgecolor=colors[0], ecolor=colors[0], elinewidth=3, capsize=0, label=r'joint fit data (5 nights)')

    if flag == 'normalized':
        # calculate median transit depth across all wavelength bins, acros all nights
        allwavefiles = np.array([loldict[night]['wavefiles'] for night in nights])
        meddepth = np.median([[loldict[night][w]['depth'] for w in allwavefiles[n]] for n,night in enumerate(nights)])*100.
        print 'median depth: ', meddepth

        if model:
            for m in modelspectra:
                model_wavelengths, model_depths, model_binned_wavelengths, model_binned_depths = np.array(m[0]), np.array(m[1]), np.array(m[2]), np.array(m[3])
                plt.plot(model_wavelengths, model_depths*scale*100./np.median(model_depths*scale*100.)*meddepth, color='k', alpha=0.5, linewidth=2, label=r'$100\%\ H_2/He$')
                plt.plot(model_wavelengths, [meddepth for i in model_wavelengths], color='k', ls='--', alpha=0.5, linewidth=2, label=r'$\mathrm{flat}$')
                #plt.plot(model_binned_wavelengths, model_binned_depths*scale*100./np.median(model_depths*scale*100.), 'ks', alpha=0.75)

        if medavg:
            alldepth = []
            alldepthunc = []

        for i, night in enumerate(nights):
            wavefiles = loldict[night]['wavefiles']
            wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

            xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
            depth =  [loldict[night][w]['depth']*100. for w in wavefiles]
            depthunc =  [loldict[night][w]['depthunc']*100. for w in wavefiles]
            if medavg:
                alldepth.append(depth)
                alldepthunc.append(depthunc)
                plt.errorbar(xaxis, depth/np.median(depth)*meddepth, yerr=depthunc/np.median(depth)*meddepth, fmt='o', markersize=8, color=colors[i], markeredgecolor=colors[i], ecolor=colors[i], elinewidth=2, capsize=0, alpha=0.25, label=night)
            else: plt.errorbar(xaxis, depth/np.median(depth)*meddepth, yerr=depthunc/np.median(depth)*meddepth, fmt='o', markersize=8, color=colors[i], markeredgecolor=colors[i], ecolor=colors[i], elinewidth=2, capsize=0, alpha=0.7, label=night)

        if medavg: 
            alldepth = np.array(alldepth)
            alldepthunc = np.array(alldepthunc)
            weighteddepth = np.sum(alldepth/alldepthunc**2, 0)/np.sum(1./alldepthunc**2, 0)
            weighteddepthunc = np.sqrt(1./np.sum(1./alldepthunc**2, 0))
            print 'inverse variance weighted depths:'
            print weighteddepth
            try: plt.errorbar(xaxis, weighteddepth/np.median(weighteddepth)*meddepth, yerr=weighteddepthunc/np.median(weighteddepth)*meddepth, fmt='o', markersize=8, color='k', markeredgecolor='k', ecolor='k', elinewidth=2, capsize=0, alpha=0.9, label='weighted average')
            except(ValueError):
                print 'check that all of your directories have the same number of wavelength bins!'
                print 'cannot take the inverse variance weighted average of differently binned points'

    plt.legend(loc=2, ncol=2)
    plt.xlim(wave_range)
    plt.ylabel('transit depth [%]', fontsize=20)
    plt.xlabel('wavelength [microns]', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    #plt.savefig('/home/hdiamond/GJ1132/transmission_spectrum_models.png', dpi=900)
    plt.show()

################################################################################################
################# plot all data ################################################################
################################################################################################

def alldata():

    plt.figure(figsize=(14, 8))

    wavefiles = loldict['joint']['wavefiles']
    wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

    xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
    depth =  [loldict['joint'][w]['depth']*100. for w in wavefiles]
    depthunc =  [np.mean(loldict['joint'][w]['depthunc']*100.) for w in wavefiles]


    z = np.polyfit(xaxis, depth, deg=1)
    p = np.poly1d(z)
    x = np.linspace(.4, 5., 1000)
    plt.plot(x, [np.median(depth) for i in x], 'k--', lw=2, alpha=0.8)#, label=r'flat fit, pvalue = {0}'.format('%.3f'%pval_flat))
    plt.plot(x, p(x), 'k:', lw=3, alpha=0.6)#, label=r'linear fit, pvalue = {0}'.format('%.3f'%pval_fit))

    plt.errorbar(xaxis, depth, yerr=depthunc, fmt='o', markersize=10, color='royalblue', markeredgecolor='royalblue', ecolor='royalblue', elinewidth=3, capsize=0, alpha=0.9, label=r'This work')

    # southworth data
    #southworth_star = 0.255 * u.solRad

    #southworth_g_wave = 0.477
    #southworth_g = np.array([1.209, 1.475, 1.567, 1.318, 1.221, 1.457, 1.515, 1.570, 1.255]) * u.earthRad
    #southworth_g_uncs = np.array([.154, .091, .151, .122, .12, .228, .146, .121, .183]) * u.earthRad
    #southworth_g_rprs = np.array((southworth_g/southworth_star).decompose())
    #southworth_g_rprs_uncs = np.array((southworth_g_uncs/southworth_star).decompose())
    #southworth_g_depths = southworth_g_rprs**2
    #southworth_g_depths_uncs = southworth_g_depths*np.sqrt(2.*((southworth_g_rprs_uncs/southworth_g_rprs)**2))
    #plt.errorbar([southworth_g_wave for i in range(len(southworth_g))], southworth_g_depths*100., yerr=southworth_g_depths_uncs*100., color='m', fmt='o', markersize=8, markeredgecolor='m', ecolor='m', elinewidth=2, capsize=0, alpha=0.2, label='Southworth')        

    southworth_wave = np.array([.477, .623, .763, .913, 1.23, 1.645, 2.165])
    southworth_rb = np.array([.00382, .00402, .00386, .00446, .00354, .00324, .00473])
    southworth_rb_unc = np.array([.00011, .00009, .00006, .00015, .00045, .00044, .00058])
    southworth_RA_a = .07733476
    southworth_rprs = southworth_rb/southworth_RA_a
    southworth_rprs_uncs = southworth_rb_unc/southworth_RA_a
    southworth_depths = southworth_rprs**2
    southworth_depths_uncs = southworth_depths*2.*(southworth_rprs_uncs/southworth_rprs)
    southworth_labels = ['g', 'r', 'i', 'z', 'J', 'H', 'K']
    plt.errorbar(southworth_wave, southworth_depths*100., yerr=southworth_depths_uncs*100., color='darkorchid', fmt='o', markersize=10, markeredgecolor='darkorchid', ecolor='darkorchid', elinewidth=3, capsize=0, alpha=0.8, label='Southworth, et al. (2017)')
    #plt.plot(southworth_wave, southworth_depths/np.median(weighted_depths), yerr=southworth_depths_uncs/np.median(weighted_depths), color='k', fmt='o', markersize=10, markeredgecolor='k', ecolor=color'k', elinewidth=3, capsize=4, alpha=0.6, label='Southworth')

    # MEarth data
    mearth_wave = 0.9
    mearth_rprs = 0.0455
    mearth_rprs_unc = 0.0006
    mearth_depth = mearth_rprs**2
    mearth_depth_unc = mearth_depth*np.sqrt(2.*((mearth_rprs_unc/mearth_rprs)**2))
    mearth_label = 'MEarth, Dittmann et al. (2017)'
    plt.errorbar(mearth_wave, mearth_depth*100., yerr=mearth_depth_unc*100., color='forestgreen', fmt='o', markersize=10, markeredgecolor='forestgreen', ecolor='forestgreen', elinewidth=3, capsize=0, alpha=0.8, label=mearth_label)
    #plt.errorbar(mearth_wave, mearth_depth/np.median(weighted_depths), yerr=mearth_depth_unc/np.median(weighted_depths), color='k', fmt='o', markersize=10, markeredgecolor='k', ecolor=color'k', elinewidth=3, capsize=4, alpha=0.6)

    # Spitzer data
    spitzer_wave = 4.5      # microns
    spitzer_rprs = 0.0492
    spitzer_rprs_unc = 0.0008
    spitzer_depth = spitzer_rprs**2
    spitzer_depth_unc = spitzer_depth*np.sqrt(2.*((spitzer_rprs_unc/spitzer_rprs)**2))
    spitzer_label = 'Spitzer, Dittmann et al. (2017)'
    plt.errorbar(spitzer_wave, spitzer_depth*100., yerr=spitzer_depth_unc*100.,  fmt='o', color='firebrick', markersize=10, markeredgecolor='firebrick', ecolor='firebrick', elinewidth=3, capsize=0, alpha=0.8, label=spitzer_label)
    #plt.errorbar(spitzer_wave, spitzer_depth/np.median(weighted_depths), yerr = spitzer_depth_unc/np.median(weighted_depths), color='k', fmt='o', markersize=10, markeredgecolor='k', ecolor=color'k', elinewidth=3, capsize=4, alpha=0.6)

    plt.legend(loc='best')
    plt.xlim(.4, 5.)
    plt.xscale('log')
    plt.xticks(np.array([.4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5]), ('0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', '2', '3', '4', '5'), fontsize=18)
    plt.yticks(fontsize=18)
    #plt.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel('transit depth [%]', fontsize=20)
    plt.xlabel('wavelength [microns]', fontsize=20)
    plt.tight_layout()
    plt.savefig('/home/hdiamond/GJ1132/transmission_spectra_alldata.png', dpi=900)
    plt.show()

##################################################################################
############################ plot parameters ####################################
##################################################################################

def parameters():

    plt.figure(figsize=(8, 18))
    gs = plt.matplotlib.gridspec.GridSpec(len(inputs.fitlabels[0])+3, 1, hspace=0.075, wspace=0.0, left=0.14,right=0.98, bottom=0.05, top=0.98)
    paramplots = {}
    paramplots['depth'] = plt.subplot(gs[0,0])
    paramplots['u0'] = plt.subplot(gs[1,0])
    paramplots['u1'] = plt.subplot(gs[2,0])
    for f, flabel in enumerate(inputs.fitlabels[0]):
        paramplots[flabel] = plt.subplot(gs[f+3,0])

    wavefiles = loldict['joint']['wavefiles']
    wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

    xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
    depth =  [loldict['joint'][w]['depth']*100. for w in wavefiles]
    depthunc =  [np.mean(loldict['joint'][w]['depthunc']*100.) for w in wavefiles]

    paramplots['depth'].errorbar(xaxis, depth, yerr=depthunc, fmt='o-', lw=2, markersize=6, color=colors[0], markeredgecolor=colors[0], ecolor=colors[0], elinewidth=2, capsize=0, alpha=0.9)
    paramplots['depth'].set_xticks([])
    paramplots['depth'].tick_params(axis='both', labelsize=12)
    paramplots['depth'].set_xlim(wave_range)
    paramplots['depth'].set_ylabel('depth [%]', fontsize=16)

    paramplots['u0'].errorbar(xaxis, loldict['joint']['parameters']['u0'], yerr=loldict['joint']['parameters']['u0unc'], fmt='o-', lw=2, markersize=6, color=colors[0], markeredgecolor=colors[0], ecolor=colors[0], elinewidth=2, capsize=0, alpha=0.9)
    paramplots['u0'].set_xticks([])
    paramplots['u0'].tick_params(axis='both', labelsize=12)
    paramplots['u0'].set_xlim(wave_range)
    paramplots['u0'].set_ylabel('u0', fontsize=16)

    paramplots['u1'].errorbar(xaxis, loldict['joint']['parameters']['u1'], yerr=loldict['joint']['parameters']['u1unc'], fmt='o-', lw=2, markersize=6, color=colors[0], markeredgecolor=colors[0], ecolor=colors[0], elinewidth=2, capsize=0, alpha=0.9)
    paramplots['u1'].set_xticks([])
    paramplots['u1'].tick_params(axis='both', labelsize=12)
    paramplots['u1'].set_xlim(wave_range)
    paramplots['u1'].set_ylabel('u1', fontsize=16)

    for flabel in inputs.fitlabels[0]:
        for n, night in enumerate(inputs.nightname):
            paramplots[flabel].errorbar(xaxis, loldict['joint']['parameters'][night][flabel], yerr=loldict['joint']['parameters'][night][flabel+'unc'], fmt='o-', lw=2, markersize=6, color=nightcolor[n], markeredgecolor=nightcolor[n], ecolor=nightcolor[n], elinewidth=2, capsize=0, alpha=0.9)
        paramplots[flabel].set_ylabel(flabel, fontsize=16)
        paramplots[flabel].set_xlim(wave_range)
        paramplots[flabel].tick_params(axis='both', labelsize=12)
        if flabel == inputs.fitlabels[0][-1]: paramplots[flabel].set_xlabel('wavelength [microns]', fontsize=16)
        else: paramplots[flabel].set_xticks([])

    #plt.yticks(fontsize=12)
    #plt.xticks(fontsize=12)

    plt.savefig('/home/hdiamond/GJ1132/parameter_vs_wavelength.png', dpi=900)
    plt.show()
    

