'''This script will will take detrended data and display the the results.'''
'''Figures are pretty unique to each data set so many of the actual plots are made right here!'''


import sys
#sys.path.append('/home/hdiamond/local/lib/python2.7/site-packages/')
sys.path.append('/h/mulan0/code/')
sys.path.append('/h/mulan0/code/craftroom')
sys.path.append('/h/mulan0/code/decorrasaurus')
from decorrasaurus.RunReader import RunReader

# if no argument given, RunReader tries to read in the most recent run from the ./run/ directory

try:
    r = RunReader(sys.argv[1])
    r.readrun()
except IndexError:
    r = RunReader()
    r.readrun()
    
#### plotting stuff #####
import matplotlib.pyplot as plt
import numpy as np
import pickle
colors = ['royalblue', 'firebrick', 'goldenrod', 'forestgreen', 'purple']
nightcolor = ['tomato', 'orange', 'lawngreen', 'aqua', 'fuchsia']


#-----------------------------------------------------------------------------------------------------------
#### for Magellan 2018B proposal ###
'''
1) test decorrelation params with 200A featureless region
2) make white lc with model fit plot
3) consider maing white lc without water bands
4) make transmission spectrum; put simple solar comp model behind it; level at which we are ruling out and how much more we could be if we had double the data\
SEND DRAFT TO COLLABORATORS TONIGHT
'''
def magellan2018B_whitelc():
# white lc
    n = 0
    w = 0
    t0 = r.results[n]['t0'][w]
    fitmodel = r.results[n]['fitmodel'][w]
    batmanmodel = r.results[n]['batmanmodel'][w]

    #plt.figure()
    #plt.plot(r.subcube[n]['bjd'][r.results[n]['binnedok'][w]]-t0, r.results[n]['lightcurve'][w]/r.results[n]['fitmodel'][w], 'o', markeredgecolor='none', alpha=0.6)
    #plt.plot(r.subcube[n]['bjd'][r.results[n]['binnedok'][w]]-t0, r.results[n]['batmanmodel'][w], 'k-', lw=2, alpha=0.6)
    #plt.xlabel('BJD-'+str(r.results[n]['t0'][w]), fontsize=16)
    #plt.ylabel('Normalized Flux', fontsize=16)
    #plt.tick_params(axis='both', labelsize=12)
    #plt.tight_layout()
    #plt.show()

    ##### time binned white light curve ###
    plt.figure()
    mainaxis = np.array(r.subcube[n]['bjd'][r.results[n]['binnedok'][w]]-t0)*24*60
    timebinedges = np.histogram(mainaxis, 100)[1]
    print('time in bin:', timebinedges[1]-timebinedges[0])
    timebinxaxis = []
    timebinweightlcdict_num = {}
    timebinweightresiddict_num = {}
    timebinweight_den = {}
    timebinlcdict = {}
    timebinresiddict = {}
    for t in range(len(timebinedges)-1):
        if t == len(timebinedges)-1: timebininds = np.where((mainaxis >= timebinedges[t]) & (mainaxis <= timebinedges[t+1]))
        else: timebininds = np.where((mainaxis >= timebinedges[t]) & (mainaxis < timebinedges[t+1]))
        timebinxaxis.append(np.median((timebinedges[t], timebinedges[t+1])))
        timebinweightlcdict_num[t] = []
        timebinweightresiddict_num[t] = []
        timebinweight_den[t] = []
        timebinlcdict[t] = []
        timebinresiddict[t] = []

    for n, night in enumerate(r.inputs.nightname):
        xaxis = np.array(r.subcube[n]['bjd'][r.results[n]['binnedok'][w]]-t0)*24*60
        lc = (r.results[n]['lightcurve'][w][r.results[n]['binnedok'][w]]/fitmodel)
        resid = (r.results[n]['lightcurve'][w][r.results[n]['binnedok'][w]]-(fitmodel*batmanmodel))*1e6
        weightlc = lc/r.results[n]['photnoiseest'][w][r.results[n]['binnedok'][w]]**2
        weightresid = resid/r.results[n]['photnoiseest'][w][r.results[n]['binnedok'][w]]**2
        weight = 1/r.results[n]['photnoiseest'][w]**2
        for t in range(len(timebinedges)-1):
            if t == len(timebinedges)-1: timebininds = np.where((xaxis >= timebinedges[t]) & (xaxis <= timebinedges[t+1]))
            else: timebininds = np.where((xaxis >= timebinedges[t]) & (xaxis < timebinedges[t+1]))
            timebinweightlcdict_num[t].append(weightlc[timebininds])
            timebinweightresiddict_num[t].append(weightresid[timebininds])
            timebinweight_den[t].append(weight[timebininds])
            timebinlcdict[t].append(lc[timebininds])
            timebinresiddict[t].append(resid[timebininds])

    timebinlc = [np.sum(np.hstack(timebinweightlcdict_num[t]))/np.sum(np.hstack(timebinweight_den[t])) for t in range(len(timebinedges)-1)]
    plt.plot(timebinxaxis, timebinlc, 'o', markeredgecolor='none') #, label='3-min time bin')
    plt.plot(xaxis, batmanmodel, 'k-', lw=2, alpha=0.4)
    #timebinresid = [np.median(np.hstack(timebinresiddict[t])) for t in range(len(timebinedges)-1)]
    #timebinresid = [np.sum(np.hstack(timebinweightresiddict_num[t]))/np.sum(np.hstack(timebinweight_den[t])) for t in range(len(timebinedges)-1)]
    #plt.plot(timebinxaxis, timebinresid, 'o', markeredgecolor='royalblue', label='RMS='+str(np.std(timebinresid))+' ppm')
    plt.xlabel('Time from Mid-Transit [min]', fontsize=16)
    plt.ylabel('Normalized Flux', fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.show()

    ###### transmission spectrum ######
    depth = np.array(r.results[0]['rp0'])**2
    depth_unc = depth*2*(np.array(r.results[0]['rp0_unc'])/np.array(r.results[0]['rp0']))
    xaxis = np.array(r.results[0]['midwave'])/10000.
    wavelims = r.results[0]['wavelims']
    wave_range = [wavelims[0][0]/10000., wavelims[-1][-1]/10000.]
    from scipy.special import gammainc
    from scipy.signal import resample

    z = np.polyfit(xaxis, depth, deg=1)
    p = np.poly1d(z)
    x = np.linspace(wave_range[0], wave_range[1], 100)
    fit_binned_depths = []
    for i in range(len(xaxis)):
        bininds = np.where((x >= wavelims[i][0]/10000.) & (x <= wavelims[i][1]/10000.))[0]
        fit_binned_depths.append(np.mean(p(x[bininds])))
    fit_binned_depths = np.array(fit_binned_depths)

    target = r.subcube[0]['target']
    expind = np.where(r.subcube[0]['airmass'] == min(r.subcube[0]['airmass']))[0][0]
    Mdwarfspec = r.subcube[0]['raw_counts'][target][expind]#/np.median(subcube[0]['raw_counts'][inputs.target[0]][inputs.targetpx[0]][200])

    chisq_flat = np.sum(np.power((depth*100. - np.array([np.median(depth*100.) for i in depth]))/np.mean(depth_unc*100.), 2.))
    print('chisq_flat:', chisq_flat)
    pval_flat = 1 - gammainc(0.5*(len(wavelims)-1), 0.5*chisq_flat)
    print('pval flat:', pval_flat)
    chisq_fit = np.sum(np.power((depth*100. - fit_binned_depths)/np.mean(depth_unc*100.), 2.))
    print('chisq_fit:', chisq_fit)
    pval_fit = 1 - gammainc(0.5*(len(wavelims)-1), 0.5*chisq_fit)

    plt.plot(x, [np.median(depth*100.) for i in x], 'k--', lw=2, alpha=0.8, label=r'flat fit, pvalue = {0}'.format('%.3f'%pval_flat))
    plt.plot(x, p(x)*100., 'k:', lw=3, alpha=0.6, label=r'linear fit, pvalue = {0}'.format('%.3f'%pval_fit))

    plt.errorbar(xaxis, depth*100., yerr=depth_unc*100., fmt='o', markersize=10, elinewidth=3, capsize=0, label=r'joint fit data (5 nights)')

    #plt.figure(figsize=(10, 5))
    #plt.errorbar(np.array(r.results[0]['midwave'])/10000., depth*100, yerr=depth_unc*100, fmt='o', markersize=10, color=colors[0], markeredgecolor=colors[0], ecolor=colors[0], elinewidth=3, capsize=0, label=r'joint fit data (5 nights)')
    plt.ylabel('Transit Depth (%)', fontsize=16)
    plt.xlabel('Wavelength (microns)', fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.show()









#-----------------------------------------------------------------------------------------


#### transmission spectrum ###

def transmission():
    plt.figure(figsize=(12,6))
    plt.errorbar(np.array(r.results[0]['midwave'])/10000., np.array(r.results[0]['rp0'])**2*100, yerr=(np.array(r.results[0]['rp0'])**2)*2*(np.array(r.results[0]['rp0_unc'])/np.array(r.results[0]['rp0']))*100, fmt='o', markersize=10, color=colors[0], markeredgecolor=colors[0], ecolor=colors[0], elinewidth=3, capsize=0, label=r'IMACS transit')
    plt.xlabel(r'Wavelength ($\mu$m)', fontsize=16)
    plt.ylabel(r'Transit Depth (%)', fontsize=16)
    plt.legend(loc=4, fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.show()

# add batman model and fit model to wavebin - for lmfit and for fullfit

#### lightcurves ###
def lightcurves(timebin=False, latexfile=False):
    plt.figure()
    nwave = len(r.subcube[0]['wavebin']['wavefiles'])

    gs = plt.matplotlib.gridspec.GridSpec(nwave, 2, hspace=0.0, wspace=0.17, left=0.07,right=0.98, bottom=0.05, top=0.98)
    lcplots = {}

    for w in range(nwave):
        lcplots[w] = {}
        lcplots[w][1] = plt.subplot(gs[w,1])
        lcplots[w][0] = plt.subplot(gs[w,0])

    for w, wave in enumerate(r.subcube[0]['wavebin']['wavefiles']):

        resid = []
        for n, night in enumerate(r.inputs.subdirectories):
            binnedok = loldict[night][wave]['binnedok']
            t0 = loldict[night]['t0']
            bjdbinned = (loldict[night]['bjd'][binnedok] - t0)*24*60
            lc = loldict[night][wave]['lc']
            fitmodel = loldict[night][wave]['fitmodel']
            batmanmodel = loldict[night][wave]['batmanmodel']
            resid.append(lc - fitmodel*batmanmodel)

            # need to deal with self.t0 changing during run; this is a secondary issue as you will likely only be plotting from runs where you have fixed self.t0

            if timebin:
                for t in range(len(timebinedges)-1):
                    if t == len(timebinedges)-1: timebininds = np.where((bjdbinned >= timebinedges[t]) & (bjdbinned <= timebinedges[t+1]))
                    else: timebininds = np.where((bjdbinned >= timebinedges[t]) & (bjdbinned < timebinedges[t+1]))
                    timebinlcdict[t].append((lc/fitmodel)[timebininds])
                    timebinresiddict[t].append(((lc-fitmodel*batmanmodel)*1e6)[timebininds])
                    lcplots[w][0].set_xlim(-73, 73)
                    lcplots[w][0].set_ylim(.996, 1.004)
                    if w == nwave/2: 
                        lcplots[w][0].set_yticks([.996, 1.0, 1.004])
                        lcplots[w][0].ticklabel_format(useOffset=False)
                    else: lcplots[w][0].set_yticks([])
                    if w == nwave-1: pass
                    else: lcplots[w][0].set_xticks([]) 
                    lcplots[w][1].set_xlim(-73, 73)
                    lcplots[w][1].set_ylim(-1500, 1500)
                    if w == nwave/2: lcplots[w][1].set_yticks([-1500, 0, 1500])
                    else: lcplots[w][1].set_yticks([])
                    if w == nwave-1: pass
                    else: lcplots[w][1].set_xticks([])


            else:
                lcplots[w][0].plot(bjdbinned, (loldict[night][wave]['lc']/loldict[night][wave]['fitmodel']), 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.5)
                lcplots[w][0].plot(bjdbinned, batmanmodel, color='k', lw=2, alpha=0.8)
                lcplots[w][0].set_xlim(-73, 73)
                lcplots[w][0].set_ylim(.993, 1.007)
                if w == nwave/2: lcplots[w][0].set_yticks([.993, 1.0, 1.007])
                else: lcplots[w][0].set_yticks([])
                if w == nwave-1: pass
                else: lcplots[w][0].set_xticks([]) 


                lcplots[w][1].plot(bjdbinned, (lc - fitmodel*batmanmodel)*1e6, 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.5)
                lcplots[w][1].axhline(0, 0.025, 1-.025, color='k', lw=2, alpha=0.8)
                lcplots[w][1].set_xlim(-73, 73)
                lcplots[w][1].set_ylim(-5000, 5000)
                if w == nwave/2: lcplots[w][1].set_yticks([-5000, 0, 5000])
                else: lcplots[w][1].set_yticks([])
                if w == nwave-1: pass
                else: lcplots[w][1].set_xticks([])
                #if w == len(wavefiles)-1: pass
                #else: lcplots[0][1].axhline(1-offset*w-0.5*offset, 0, 1, color='k', lw=1)

            #lcplots[0][0].set_xlim(-0.05, 0.05)

        if timebin: 
            timebinlc = [np.median(np.hstack(timebinlcdict[t])) for t in range(len(timebinedges)-1)]
            lcplots[w][0].plot(timebinxaxis, timebinlc, 'o', color='royalblue', markeredgecolor='royalblue')
            lcplots[w][0].plot(bjdbinned, batmanmodel, color='k', lw=2, alpha=0.8)
            timebinresid = [np.median(np.hstack(timebinresiddict[t])) for t in range(len(timebinedges)-1)]
            lcplots[w][1].plot(timebinxaxis, timebinresid, 'o', color='royalblue', markeredgecolor='royalblue')
            lcplots[w][1].axhline(0, 0.025, 1-.025, color='k', lw=2, alpha=0.8)


        if timebin:
            rms = np.std(timebinresid)/1e6
            lcplots[w][0].text(-69, 1.0015, wave+' A', color='k', fontsize=14)   #'#ff471a'
        else: 
            allresid = np.hstack(resid)
            rms = np.std(allresid)
            lcplots[w][0].text(-69, 1.004, wave+' A', color='k', fontsize=14)   #'#ff471a'
        #lcplots[w][1].text(-0.048, 2250, 'RMS: {0:.0f} ppm'.format(rms*1e6), color='k', fontsize=14)
        #lcplots[0][0].text(0.0125, 1.0025-offset*k, 'x exp. noise: {0:.2f}'.format(mcmcRMS/expnoise), color='#ff471a', weight='bold', fontsize=12)
        if latexfile: 
            texfile = open('/home/hdiamond/GJ1132/transit_depths_tabular.txt', 'a')
            binrange = str(int(float(wave.split('-')[0])))+'-'+str(int(float(wave.split('-')[1])))
            texfile.write(binrange + ' & ' + str(int(round(rms*1e6))) + ' & ')
            texfile.write('%.3f'%(loldict['joint'][wave]['depth']*100.))
            texfile.write(' \pm ')
            try: texfile.write('%.3f'%np.mean([loldict['joint'][wave]['depthunc'][0]*100., loldict['joint'][wave]['depthunc'][1]*100.]))
            except: texfile.write('%.3f'%(loldict['joint'][wave]['depthunc']*100.))
            texfile.write(' & ')
            texfile.write('%.2f'%(loldict['joint'][wave]['timesexpnoise']))
            texfile.write('\\\\ \n')


    lcplots[nwave-1][0].set_xlabel('time from mid-transit [min]', fontsize=15)
    lcplots[nwave-1][0].tick_params(axis='x', labelsize=12)

    lcplots[nwave/2][0].set_ylabel('normalized flux', fontsize=15)
    lcplots[nwave/2][0].tick_params(axis='y', labelsize=12)
    #plt.ylim(0.9955, 1.002)
    #lcplots[8][0].set_ylim(.84, 1.006)
    #plt.suptitle('wavelength range: ' + str(loldict['joint']['wavelims']) + ' A, binsize: ' + str(loldict['joint']['binlen']) + ' A', fontsize=15)
    lcplots[nwave-1][1].set_xlabel('time from mid-transit [min]', fontsize=15)
    lcplots[nwave-1][1].tick_params(axis='x', labelsize=12)

    lcplots[nwave/2][1].set_ylabel('residuals (ppm)', fontsize=15)
    lcplots[nwave/2][1].tick_params(axis='y', labelsize=12)
    #lcplots[0][1].set_yticks([.915, .92, .925])#np.array([.995, 1., 1.005]), ('-2500', '0', '2500'))
    #lcplots[0][1].set_yticklabels(['-2500', '0', '2500'])
    #lcplots[0][1].set_yticks(np.array([.995, 1., 1.005]), ('-2500', '0', '2500'))
    #plt.ylim(0.9955, 1.002)
    #lcplots[0][1].set_ylim(.84, 1.006)    

    if latexfile:
        texfile.write('\\hline \n')
        texfile.write('\\end{tabular} \n')
        texfile.close()

    if timebin: plt.savefig('/home/hdiamond/GJ1132/bin200A_lightcurves_joint_timebin.png')
    else: plt.savefig('/home/hdiamond/GJ1132/bin200A_lightcurves_joint.png')
    plt.show()

