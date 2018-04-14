from imports import *
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

        self.speak('run has been read in')
    
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

        results = []

        # want array of depths and uncertainties

        # only need to read stuff in from the wavebin .npy files; don't bother putting subcube stuff inthere - it's aready loaded in!
        for r in self.subdirectories:
            result = {}
            for w in self.subcube['wavebin']['wavefiles']:
                binnedresult = np.load(self.rundirectory+w+'.npy'])[()]





