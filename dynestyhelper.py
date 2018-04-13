import numpy as np
from ModelMakerJoint import ModelMakerJoint
from ModelMaker import ModelMaker
import multiprocessing
import dynesty


def lnlike(p, detrender, inputs, wavebin):
    if detrender.joint: 
        modelobj = ModelMakerJoint(inputs, wavebin, p)
        models = modelobj.makemodel()
        logl = []
        for n, night in enumerate(inputs.nightname):
            # p[sind] is an 's' parameter; if the uncertainties do not need to be re-scaled then s = 1
            # there is a single 's' parameter for each night's fit - helpful if a dataset is far from the photon noise
            sind = int(np.where(np.array(inputs.freeparamnames) == 's'+str(n))[0])
            penaltyterm = -len(wavebin['photnoiseest'][n]) * np.log(p[sind])
            chi2 = ((wavebin['lc'][n] - models[n])/wavebin['photnoiseest'][n])**2
            logl.append(penaltyterm - 0.5*(1./(p[sind]**2))*np.sum(chi2))
        return np.sum(logl)
    else:
        modelobj = ModelMaker(inputs, wavebin, p)
        model = modelobj.makemodel()

        # p[sind] is an 's' parameter; if the uncertainties do not need to be re-scaled then s = 1
        # there is a single 's' parameter for each night's fit - helpful if a dataset is far from the photon noise
        sind = int(np.where(np.array(inputs.freeparamnames) == 's')[0])
        penaltyterm = -len(wavebin['photnoiseest']) * np.log(p[sind])
        chi2 = ((wavebin['lc'] - model)/wavebin['photnoiseest'])**2
        logl = penaltyterm - 0.5*(1./(p[sind]**2))*np.sum(chi2)
        return logl

def ptform(p, mcmcbounds):
    x = np.array(p)
    for i in range(len(x)):
        span = mcmcbounds[1][i] - mcmcbounds[0][i]
        x[i] = x[i]*span + mcmcbounds[0][i]
    return x

def dyn(detrender, inputs, wavebin, mcmcbounds, ndim):
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, bound='multi', sample='slice', update_interval=float(ndim), pool=pool, logl_kwargs={'detrender':detrender, 'inputs':inputs, 'wavebin':wavebin}, ptform_kwargs={'mcmcbounds':mcmcbounds})
    dsampler.run_nested(nlive_init=int(5*ndim), nlive_batch=int(5*ndim), wt_kwargs={'pfrac': 1.0}) # place 100% of the weight on the posterior, don't sample the evidence
    return dsampler

