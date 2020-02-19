from .imports import *
import dynesty
from dynesty.dynamicsampler import stopping_function, weight_function
from dynesty.plotting import _quantile
from dynesty import utils as dyfunc
import sys
from multiprocessing import Pool
from scipy import stats
import scipy.interpolate as interpolate
import dill as pickle
import numpy as np

sys.path.append('/pool/starbuck0/code/')
sys.path.append('/pool/starbuck0/code/craftroom')
sys.path.append('/pool/starbuck0/code/decorrasaurus/decorrasaurus')

class dynestyhelper(Talker):
    '''RunReader takes your decorrasaurus run and amalgamates the results'''

    def __init__(self, *rundirectory):
        '''initialize from an input.init file'''

        Talker.__init__(self)

        if rundirectory: 
            
            self.rundirectory = str(rundirectory[0])
            self.speak('the dynesty helper is starting in {0}'.format(self.rundirectory))

        else: 
            rundirectories = [d for d in os.listdir('./run/') if os.path.isdir(os.path.join('./run/', d))]
            rundirectories = sorted(rundirectories)#, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))
            self.rundirectory = './run/'+rundirectories[-1]+'/'
            self.speak('the dynesty helper is using the last run in directory {0}'.format(self.rundirectory))

        #return self.rundirectory




'''
#testdict = np.load('/pool/starbuck0/analysis/LHS1140/LDSS3C2018/run/2020-02-14-11:08_binall-georgeGP-testInputs/6700.0-10100.0test.npy', allow_pickle=True)[()]
global rangeofdirectories
global lcs
global binnedok
global mcmcbounds
global kernelinds
global freeparamnames
global inputs
global wavebin
global span
global gps

def lnlike(p, gps, modelobj, rangeofdirectories):  
    [gps[i].set_parameter_vector(np.array(p)[modelobj.allparaminds[i]]) for i in rangeofdirectories]
    return np.sum([gps[i].log_likelihood(lcs[i][binnedok[i]]) for i in rangeofdirectories])

def ptform(p, span, mcmcbounds, expspan, expboundslo, kernelinds):

    x = np.array(p)
    x = x*span + mcmcbounds[0]

    loguniformdist = [np.log(p[i]*expspan[i] + expboundslo[i]) for i in kernelinds]
    x[kernelinds] = loguniformdist

    return x

def rundynesty(testdict):

    rangeofdirectories = testdict['rangeofdirectories']
    lcs = testdict['lcs']
    binnedok = testdict['binnedok']
    #mcmcbounds = testdict['mcmcbounds']
    kernelinds = testdict['kernelinds']
    #kernelinds = [4, 5, 6]
    freeparamnames = testdict['freeparamnames']
    inputs = testdict['inputs']
    wavebin = testdict['wavebin']

    ndim = len(freeparamnames)

    modelobj = ModelMaker(inputs, wavebin)
    gps = modelobj.makemodelGP()

    mcmcbounds = np.concatenate([np.array(gps[i].get_parameter_bounds()) for i in rangeofdirectories]).T
    span = mcmcbounds[1] - mcmcbounds[0]

    expboundslo = np.exp(mcmcbounds[0])
    expspan = np.exp(mcmcbounds[1]) - np.exp(mcmcbounds[0])


    pool = Pool(processes=4)

    dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, pool=pool, queue_size=8, logl_kwargs={'gps':gps, 'modelobj':modelobj, 'rangeofdirectories':rangeofdirectories}, ptform_kwargs={'span':span, 'mcmcbounds':mcmcbounds, 'expspan':expspan, 'expboundslo':expboundslo, 'kernelinds':kernelinds})
    #dsampler.run_nested(wt_kwargs={'pfrac': 1.0})

    return dsampler


'''
