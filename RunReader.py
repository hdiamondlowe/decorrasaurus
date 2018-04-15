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
        self.subdirectories = sorted(self.subdirectories, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))

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
            for f in self.inputs.fitlabels[n]: result[f+str(n)] = []
            for f in self.inputs.fitlabels[n]: result[f+str(n)+'_unc'] = []
            result['dt'+str(n)] = []
            result['dt'+str(n)+'_unc'] = []
            result['rp'+str(n)] = []
            result['rp'+str(n)+'_unc'] = []
            result['wavelims'] = []
            result['midwave'] = []
            result['lightcurve'] = []
            result['binnedok'] = []
            result['fitmodel'] = []
            result['batmanmodel'] = []

            # only want to have to read in wavebins once
            # something will have to happen in here when two instruments cover different wavebands
            for w in self.subcube[n]['wavebin']['wavefiles']:
                binnedresult = np.load(self.rundirectory+w+'.npy')[()]
                for i, p in enumerate(binnedresult['freeparams']):
                    result[p].append(binnedresult['lmfit']['values'][i])
                    result[p+'_unc'].append(binnedresult['lmfit']['uncs'][i])
                result['wavelims'].append(binnedresult['wavelims'])
                result['midwave'].append(np.mean(binnedresult['wavelims']))
                result['lightcurve'].append(binnedresult['lc'])
                result['binnedok'].append(binnedresult['binnedok'][n])
                result['fitmodel'].append(binnedresult['fitmodel'][n])
                result['batmanmodel'].append(binnedresult['batmanmodel'][n])
                    
            self.results.append(result)

        self.speak('run has been read in')

        # needed for transmission spec
        # rp/rs, rp/rs unc, wavelims

        # needed for lightcurves
        # what the free parameters are that went into the fit; inlcuding reparameterized limb darkening
        # batman model and lightcurve model
        # binnedok - for masking correct bjd point
        

