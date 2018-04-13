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

        self.directories = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.', d))]
        if 'run' in self.directories: self.directories.remove('run')
        self.directories = sorted(self.directories, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))

        try: 
            self.inputs = Inputs(self.directories, self.directoryname)
        except(AttributeError): 
            self.inputs = Inputs(self.directories)
            self.directoryname = self.inputs.directoryname

        # try first loading in a saved minicube; if it doesn't exist read in the whole original cube
        self.cube = CubeReader(self, self.directories)

        # try first to look for the files with the lcs already in them, eg. '7000.npy'
        self.lcs = LCMaker(self, self.directories)
    
    def detrend(self):
        
        self.speak('detrending data from nights {0} in directory {1}'.format(self.inputs.nightname, self.directoryname))

        for w, wavefile in enumerate(self.lcs.wavebin.wavefiles):
            if self.inputs.fixedrp:
                rpind = np.where(np.array(self.inputs.tranlabels) == 'rp')[0][0]
                self.inputs.tranparams[rpind] = self.inputs.fixedrp[w]
                Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')
                self.speak('lmfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                self.write('lmfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
            self.lmfit = LMFitter(self, wavefile)

        if self.inputs.domcmc:
            for wavefile in self.lcs.wavebin.wavefiles:
                if self.inputs.fixedrp:
                    rpind = np.where(np.array(self.inputs.tranlabels) == 'rp')[0][0]
                    self.inputs.tranparams[rpind] = self.inputs.fixedrp[w]
                    Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')
                    self.speak('mcfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                    self.write('mcfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                self.mcfit = MCFitter(self, wavefile)


        self.speak('decorrelation complete')
        self.speak('decorrasaurus has done all it can do for your data. goodbye!')
