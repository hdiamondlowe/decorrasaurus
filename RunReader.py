from .imports import *
import astrotools.modeldepths as md
import astropy.units as u
from .ModelMaker import ModelMaker
from .Inputs import Inputs
import collections
from datetime import datetime
from ldtk import LDPSetCreator, BoxcarFilter
import pickle
from scipy.signal import resample
from scipy.special import gammainc

class RunReader(Talker, Writer):
    '''Detrenders are objects for detrending data output by mosasaurus.'''

    def __init__(self, *rundirectory):
        '''initialize from an input.init file'''

        # decide whether or not this Reducer is chatty
        Talker.__init__(self)

        if rundirectory: 
            
            self.rundirectory = str(rundirectory[0])
            self.speak('the run reader is starting in directory {0}'.format(self.rundirectory))

        else: 
            rundirectories = [d for d in os.listdir('./run/') if os.path.isdir(os.path.join('./run/', d))]
            rundirectories = sorted(rundirectories)#, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))
            self.rundirectory = './run/'+rundirectories[-1]+'/'
            self.speak('the run reader is using the last run in directory {0}'.format(self.rundirectory))

        # get set up to read the run in
        self.setup()
        self.speak('run has been set up')
    
    def setup(self):
        # read in inputs

        # need the directory names as an input to Inputs
        self.subdirectories = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.', d))]
        if 'run' in self.subdirectories: self.subdirectories.remove('run')
        if 'notinuse' in self.subdirectories: self.subdirectories.remove('notinuse')
        #self.subdirectories = sorted(self.subdirectories, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))
        self.subdirectories = [1]

        self.inputs = Inputs(self.subdirectories, self.rundirectory)

        self.subcube = np.load(self.rundirectory+'subcube.npy')[()]

    def readrun(self):
        # create and populate a dictionary of run information
        self.speak('reading in run')
        self.results = []

        # want array of depths and uncertainties

        # only need to read stuff in from the wavebin .npy files; don't bother putting subcube stuff inthere - it's aready loaded in!
        for n, subdir in enumerate(self.subdirectories):
            result = {}
            # initiate dictionarly with freeparamnames
            # need a way better way to do this - should have free parameters saved by night, not just some crazy long list of all free paramanames; more dictionaries...
            for f in self.inputs.freeparamnames: result[f] = []
            for f in self.inputs.freeparamnames: result[f+'_unc'] = []
            result['wavelims'] = []
            result['midwave'] = []
            result['lightcurve'] = []
            result['ldparams'] = {}
            result['ldparams']['v0'] = []
            result['ldparams']['v1'] = []
            result['ldparams']['v0_unc'] = []
            result['ldparams']['v1_unc'] = []
            result['binnedok'] = []
            result['fitmodel'] = []
            result['batmanmodel'] = []
            result['t0'] = []    
            result['photnoiseest'] = []


            # only want to have to read in wavebins once
            # something will have to happen in here when two instruments cover different wavebands
            for w in self.subcube[n]['wavebin']['wavefiles']:
                binnedresult = np.load(self.rundirectory+w+'.npy')[()]
                # this will not work when multiplt nights are involved - need to some how separate back into each night's values
                if 'mcfit' in binnedresult.keys(): fit = 'mcfit'
                else: fit = 'lmfit'
                result['wavelims'].append(binnedresult['wavelims'])
                result['midwave'].append(np.mean(binnedresult['wavelims']))
                # these are night- and wave-dependent; only want 1 night in there at a time
                result['lightcurve'].append(binnedresult['lc'][n])
                if fit == 'lmfit':
                    for i, p in enumerate(binnedresult['freeparams']):
                        result[p].append(binnedresult[fit]['values'][i])
                        result[p+'_unc'].append(binnedresult[fit]['uncs'][i])
                    result['ldparams']['v0'].append(binnedresult['ldparams']['v0'])
                    result['ldparams']['v1'].append(binnedresult['ldparams']['v1'])
                    result['ldparams']['v0_unc'].append(binnedresult['ldparams']['v0_unc'])
                    result['ldparams']['v1_unc'].append(binnedresult['ldparams']['v1_unc'])
                elif fit == 'mcfit':
                    for i, p in enumerate(binnedresult['freeparams']):
                        result[p].append(binnedresult[fit]['values'][i])
                        result[p+'_unc'].append(np.mean(binnedresult[fit]['uncs'][i]))
                    result['ldparams']['v0'].append(binnedresult[fit]['values'][-3])
                    result['ldparams']['v1'].append(binnedresult[fit]['values'][-2])
                    result['ldparams']['v0_unc'].append(np.mean(binnedresult[fit]['values'][-3]))
                    result['ldparams']['v1_unc'].append(np.mean(binnedresult[fit]['values'][-2]))
                result['binnedok'].append(binnedresult['binnedok'][n])
                result['fitmodel'].append(binnedresult['lmfit']['fitmodels'][n])
                result['batmanmodel'].append(binnedresult['lmfit']['batmanmodels'][n])
                result['photnoiseest'].append(binnedresult['photnoiseest'][n])

                if 'dt'+str(n) in binnedresult['freeparams']:
                    #dtind = int(np.where(np.array(binnedresultfreeparamnames) == 'dt'+str(n))[0])
                    result['t0'].append(self.inputs.toff[n] + result['dt'+str(n)][-1])
                else:
                    dtind = int(np.where(np.array(self.inputs.tranlabels[n]) == 'dt')[0])
                    result['t0'].append(self.inputs.toff[n] + self.inputs.tranparams[n][dtind])
                    
            self.results.append(result)

        self.speak('run has been read in')

        # needed for transmission spec
        # rp/rs, rp/rs unc, wavelims

        # needed for lightcurves
        # what the free parameters are that went into the fit; inlcuding reparameterized limb darkening
        # batman model and lightcurve model
        # binnedok - for masking correct bjd point
        

