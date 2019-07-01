import numpy as np
import analysis.BatmanLC as BLC
import emcee
from emcee.utils import MPIPool
from ModelMaker import ModelMaker
import gc
# This awkward and kludge-y code exists because emcee is weird about multiprocessing stuff from within a class; now this emcee code stands alone, on its own terms, and can multiprocess in peace. We salute you.

def get_variables(bjd, binnedok, toff, fitlabels, paramlabels, tranlabels, tranparams, fit_params_minidict, mcmcbounds, istarget, isasymm, cleanup=False):

    if cleanup == False:
        global bjd_mc
        global binnedok_mc
        global toff_mc
        global fitlabels_mc
        global paramlabels_mc
        global tranlabels_mc
        global tranparams_mc
        global fit_params_minidict_mc
        global mcmcbounds_mc
        global istarget_mc
        global isasymm_mc

        bjd_mc = []
        binnedok_mc = []
        toff_mc = 0.
        fitlabels_mc = []
        paramlabels_mc = []
        tranlabels_mc = []
        tranparams_mc = []
        fit_params_minidict_mc = {}
        mcmcbounds_mc = [[],[]]
        istarget_mc = False
        isasymm_mc = False

        print 'importing parameters'

        for b in bjd:
            bjd_mc.append(b)
        for ok in binnedok:
            binnedok_mc.append(ok)    
        for f in fitlabels:
            fitlabels_mc.append(f)
        for p in paramlabels:
            paramlabels_mc.append(p)
        for t in tranlabels:
            tranlabels_mc.append(t)
        for t in tranparams:
            tranparams_mc.append(t)
        for m in mcmcbounds[0]:
            mcmcbounds_mc[0].append(m)
        for m in mcmcbounds[1]:
            mcmcbounds_mc[1].append(m)
        fit_params_minidict_mc = fit_params_minidict.copy()
        istarget_mc = istarget
        isasymm_mc = isasymm
        toff_mc = toff

        bjd_mc = np.array(bjd_mc)
        binnedok_mc = np.array(binnedok_mc)

    elif cleanup == True:

        del bjd_mc, binnedok_mc, toff_mc, fitlabels_mc, paramlabels_mc, tranlabels_mc, tranparams_mc, fit_params_minidict_mc, mcmcbounds_mc, istarget_mc, isasymm_mc
        gc.collect()

def make_model(param_values):
    x = []
    for f in range(len(fitlabels_mc)):
        x.append(param_values[f]*fit_params_minidict_mc[fitlabels_mc[f]][0])
    fit_model = np.sum(x, 0) + 1

    tranvalues = {}
    for t in range(len(tranlabels_mc)):
        if tranlabels_mc[t] in paramlabels_mc:
            ind = np.where(np.array(paramlabels_mc) == tranlabels_mc[t])[0]
            if tranlabels_mc[t] == 'q0':
                q0, q1 = param_values[ind], param_values[ind + 1]
                tranvalues['u0'] = 2.*np.sqrt(q0)*q1
                tranvalues['u1'] = np.sqrt(q0)*(1 - 2.*q1)
            elif tranlabels_mc[t] == 'q1': continue    
            else: tranvalues[tranlabels_mc[t]] = param_values[ind]
        else: 
            tranvalues[tranlabels_mc[t]] = tranparams_mc[t]
    
    if istarget_mc == True and isasymm_mc == False:
        batman = BLC.BatmanLC(bjd_mc[binnedok_mc]-toff_mc, tranvalues['dt'], tranvalues['rp'], tranvalues['per'], tranvalues['b'], tranvalues['a'], tranvalues['ecc'], tranvalues['u0'], tranvalues['u1'])
        batman_model = batman.batman_model()
    if istarget_mc == True and isasymm_mc == True:
        rp, tau0, tau1, tau2 = [], [], [], []
        numtau = 0
        for k in tranvalues.keys():
            if 'tau' in k: numtau += 1
        numdips = numtau/3
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
        for i in range(numdips):
            rp.append(tranvalues['rp'+alphabet[i]])
            tau0.append(tranvalues['tau0'+alphabet[i]])
            tau1.append(tranvalues['tau1'+alphabet[i]])
            tau2.append(tranvalues['tau2'+alphabet[i]])
        t, F = bjd_mc[binnedok_mc]-toff_mc-tranvalues['dt'], tranvalues['F']
        for i in range(len(tau0)):
            F -= 2.*rp[i] * (np.exp((t-tau0[i])/tau2[i]) + np.exp(-(t-tau0[i])/tau1[i]))**(-1)
        batman_model = F
    elif istarget_mc == False: 
        batman_model = np.ones(len(bjd_mc[binnedok_mc]))
    return fit_model*batman_model

def lnlike(p, lcb, inputs, wavebin):
    print 'i got here lnlike'
    modelobj = ModelMaker(inputs, wavebin, p)
    model = modelobj.makemodel()
    data_unc = np.power(np.std(lcb - model), 2.)
    return -0.5*np.sum(np.power((lcb-model), 2.)/data_unc + np.log(2.*np.pi*data_unc))

def lnprior(p, mcfit):
    print 'i got here lnprior'
    for i in range(len(p)):
        if not (mcfit.mcmcbounds[0][i] <= p[i] <= mcfit.mcmcbounds[1][i]):
            return -np.inf
    return 0.0

def lnprobfn(p, lcb, mcfit, inputs, wavebin, llvalues):
    print 'i got here lnprob'
    lp = lnprior(p, mcfit)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(p, lcb, inputs, wavebin)
    llvalues.append(ll)
    return lp + ll

def runemcee(mcfit, inputs, cube, wavebin):
    #bjd, binnedok, toff, fitlabels, paramlabels, tranlabels, tranparams, fit_params_minidict, mcmcbounds, istarget, isasymm, param_values, unc_values, ndim, nwalkers, nsteps, lcbinned):

    llvalues = []
    print 'i got here 1'
    ndim = len(mcfit.paramvals)
    pos = [mcfit.paramvals + mcfit.paramuncs*1e-4*np.random.randn(ndim) for i in range(inputs.nwalkers)]
    print 'i got here 2'
    sampler = emcee.EnsembleSampler(inputs.nwalkers, ndim, lnprobfn, args=(wavebin['lc'], mcfit, inputs, wavebin, llvalues))
    print 'i got here 3'

    s = sampler.sample(pos, inputs.nsteps)
    print pos, inputs.nsteps, s
    print lnprobfn

    #for i, result in enumerate(s):
    #    print i

    sampler.run_mcmc(pos, inputs.nsteps)


    return sampler, llvalues

