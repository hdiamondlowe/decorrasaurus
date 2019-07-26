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
    '''RunReader takes your decorrasaurus run and amalgamates the results'''

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
        # need the directory names as an input to Inputs
        self.subdirectories = []
        for file in os.listdir(self.rundirectory):
            if file.endswith('_input.init'):
                self.subdirectories.append(file[:-11])
        self.subdirectories = sorted(self.subdirectories)

        self.inputs = Inputs(self.subdirectories, self.rundirectory)
        self.inputs = self.inputs.inputs

        self.subcube = np.load(self.rundirectory+'subcube.npy')[()]

    def readrun(self):
        # create and populate a dictionary of run information
        self.speak('reading in run')
        self.results = {}

        # get a concatenated list of wavefiles and all possible fitted values
        self.results['allwavefiles'] = sorted(list(set(np.concatenate(([self.subcube[subdir]['wavefiles'] for subdir in self.inputs['subdirectories']])))))
        self.results['allwavelims'] = []
        self.results['allmidwave'] = []
        self.results['allparamnames'] = []

        self.results['ldparams'] = {}
        self.results['ldparams']['v0'] = []
        self.results['ldparams']['v1'] = []
        self.results['ldparams']['v0_unc'] = []
        self.results['ldparams']['v1_unc'] = []

        self.results['subwavelims'] = {}
        self.results['submidwave'] = {}
        self.results['lc'] = {}
        self.results['binnedok'] = {}
        self.results['fitmodel'] = {}
        self.results['batmanmodel'] = {}
        self.results['t0'] = {}
        self.results['photnoiseest'] = {}

        for wfile in self.results['allwavefiles']:
            # access the information for a single wavelength bin
            wavebin = np.load(self.rundirectory+wfile+'.npy')[()]

            if wavebin['mcfitdone']: fit = 'mcfit'
            elif wavebin['lmfitdone']: fit = 'lmfit'
            else: 
                print('Error: this wavelenth bin has not been fit')
                break
            self.speak('using {0} for wavelength bin {1}'.format(fit, wfile))

            self.results['allwavelims'].append(wavebin['wavelims'])
            self.results['allmidwave'].append(np.mean(wavebin['wavelims']))

            firstdir = wavebin['subdirectories'][0]
            firstn = self.inputs[firstdir]['n']

            for i, fitparam in enumerate(wavebin[fit]['freeparamnames']):
                if fitparam[:-1] in self.inputs['jointparams']: fitparam = fitparam[:-1]
                if fitparam not in self.results['allparamnames']: self.results['allparamnames'].append(fitparam)
                self.results.setdefault(fitparam, []).append(wavebin[fit]['values'][i])
                self.results.setdefault(fitparam+'_unc', []).append(wavebin[fit]['uncs'][i])
                #print(wfile, fitparam)

            if fit == 'lmfit':
                self.results['ldparams']['v0'].append(wavebin['ldparams']['v0'])
                self.results['ldparams']['v1'].append(wavebin['ldparams']['v1'])
                self.results['ldparams']['v0_unc'].append(wavebin['ldparams']['v0_unc'])
                self.results['ldparams']['v1_unc'].append(wavebin['ldparams']['v1_unc']) 
            elif fit == 'mcfit':
                i = np.argwhere(wavebin['freeparamnames'] == 'u0'+firstn)[0][0]
                self.results['ldparams']['v0'].append(wavebin[fit]['values'][i])
                self.results['ldparams']['v1'].append(wavebin[fit]['values'][i+1])
                self.results['ldparams']['v0_unc'].append(np.mean(wavebin[fit]['uncs'][i]))
                self.results['ldparams']['v1_unc'].append(np.mean(wavebin[fit]['uncs'][i+1]))

            for subdir in wavebin['subdirectories']:
                self.results['subwavelims'].setdefault(subdir,[]).append(wavebin['wavelims'])
                self.results['submidwave'].setdefault(subdir,[]).append(np.mean(wavebin['wavelims']))
                self.results['lc'].setdefault(subdir, []).append(wavebin[subdir]['lc'])
                self.results['binnedok'].setdefault(subdir,[]).append(wavebin[subdir]['binnedok'])
                self.results['fitmodel'].setdefault(subdir,[]).append(wavebin[fit]['fitmodels'][subdir])
                self.results['batmanmodel'].setdefault(subdir,[]).append(wavebin[fit]['batmanmodels'][subdir])
                self.results['photnoiseest'].setdefault(subdir,[]).append(wavebin[subdir]['photnoiseest'])

                if 'dt'+self.inputs[subdir]['n'] in wavebin[fit]['freeparamnames']:
                    dtind = np.argwhere(np.array(wavebin[fit]['freeparamnames']) == 'dt'+self.inputs[subdir]['n'])[0][0]
                    self.results['t0'].setdefault(subdir, []).append(self.inputs[subdir]['toff'] + wavebin[fit]['values'][dtind])
                else:
                    dtind = np.argwhere(np.array(self.inputs[subdir]['tranlabels']) == 'dt')[0][0]
                    self.results['t0'].setdefault(subdir, []).append(self.inputs[subdir]['toff'] + self.inputs[subdir]['tranparams'][dtind])


