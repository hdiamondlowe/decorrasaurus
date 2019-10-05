from .imports import *
from .Inputs import Inputs
from .CubeReader import CubeReader
from .LCMaker import LCMaker
from .LMFitter import LMFitter
from .FullFitter import FullFitter
from datetime import datetime

class Detrender(Talker, Writer):
    '''Detrenders are objects for detrending data output by mosasaurus.'''

    def __init__(self, *directoryname):
        '''initialize from an input.init file'''

        # decide whether or not this Reducer is chatty
        Talker.__init__(self)

        if directoryname: 
            self.speak('the detrender is starting in the folder {0}'.format(directoryname[0]))
            self.directoryname = str(directoryname[0])
        else: self.speak('making a new detrender')

        # setup all the components of the detrender
        self.setup()

        self.speak('detrender is ready to detrend')
    
    def setup(self):

        # load in the input parameters from multiple nights of data from input.init in each night's folder

        self.subdirectories = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.', d))]
        if 'run' in self.subdirectories: self.subdirectories.remove('run')
        if 'notinuse' in self.subdirectories: self.subdirectories.remove('notinuse')
        self.subdirectories = sorted(self.subdirectories)

        try: 
            self.inputs = Inputs(self.subdirectories, self.directoryname)
        except(AttributeError): 
            self.inputs = Inputs(self.subdirectories)
            self.directoryname = self.inputs.directoryname

        # try first loading in a saved minicube; if it doesn't exist read in the whole original cube
        self.cube = CubeReader(self.inputs.inputs, self.subdirectories)

        # try first to look for the files with the lcs already in them, eg. '7000.npy'
        self.lcs = LCMaker(self, self.subdirectories)
    
    def detrend(self):
        
        self.speak('detrending data from {0} in directory {1}'.format(self.subdirectories, self.directoryname))

        for w, wavefile in enumerate(self.lcs.allwavefiles):
            self.lmfit = LMFitter(self, wavefile)

        if self.inputs.inputs['fullsample']:
            for wavefile in self.lcs.allwavefiles:
                self.mcfit = FullFitter(self, wavefile)


        self.speak('decorrelation complete')
        self.speak('decorrasaurus has done all it can do for your data. goodbye!')
