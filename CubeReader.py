#import zachopy.Talker
#Talker = zachopy.Talker.Talker
#import numpy as np
from .imports import *
#import collections
from copy import deepcopy

class CubeReader(Talker):
    ''' Reads in all the datacubes from the local subdirectories'''
    def __init__(self, detrender, directories):

        Talker.__init__(self)
        
        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.directories = directories

        try: 
            self.speak('trying to read in subcube')
            self.subcube = np.load(self.inputs.saveas+'_subcube.npy')[()]
            self.speak('loaded in subcube') 
        except(IOError):
            self.speak('subcube does not exist, creating a new one')
            self.subcube = []
            for n, subdir in enumerate(self.directories):
                self.n = n
                self.subdir = subdir
                self.datacubepath = self.inputs.datacubepath[self.n]
                self.makeSubCube()
            np.save(self.inputs.saveas+'_subcube.npy', self.subcube)
            self.speak('subcube saved')

    def makeSubCube(self):

        self.speak('reading in datacube from {0} and extracting the arrays you care about'.format(self.subdir))
        cube = np.load(self.datacubepath)[()]

        self.speak('making subcube of just the arrays you care about from the full cube from {0}'.format(self.subdir))
        
        subcube = {}

        subcube['target'] = cube['target']
        subcube['comparisons'] = cube['comparisons']

        subcube['ok'] = deepcopy(cube['temporal']['ok'])               # (time)
        subcube['bjd'] = deepcopy(cube['temporal']['bjd'])             # (time)
        subcube['airmass'] = deepcopy(cube['temporal']['airmass'])     # (time)
        subcube['rotangle'] = deepcopy(cube['temporal']['rotatore'])   # (time)

        subcube['norm'] = np.ones(len(subcube['bjd']))                            # normalization constant (time)

        subcube['wavelengths'] = cube['spectral']['wavelength']        # (wave)

        # have to re-make these dictionaries
        subcube['centroid'] = deepcopy(cube['squares']['centroid'])    # [star](time)
        subcube['width'] = deepcopy(cube['squares']['width'])          # [star](time)
        subcube['stretch'] = deepcopy(cube['squares']['stretch'])      # [star](time)
        subcube['shift'] = deepcopy(cube['squares']['shift'])          # [star](time)

        subcube['raw_counts'] = deepcopy(cube['cubes']['raw_counts'])  # [star](time, wave)
        subcube['sky'] = deepcopy(cube['cubes']['sky'])                # [star](time, wave)
        subcube['dcentroid'] = deepcopy(cube['cubes']['centroid'])     # [star](time, wave)
        subcube['dwidth'] = deepcopy(cube['cubes']['width'])           # [star](time, wave)
        subcube['peak'] = deepcopy(cube['cubes']['peak'])              # [star](time, wave)

        '''
        subcube['ok'] = ok
        subcube['bjd'] = bjd
        subcube['airmass'] = airmass
        subcube['rotangle'] = rotangle
        subcube['norm'] = norm
        subcube['wavelengths'] = cube['spectral']['wavelength']       # (wave)
        
        # have to re-make these dictionaries
        subcube['centroid'] = collections.defaultdict(dict)
        subcube['width'] = collections.defaultdict(dict)
        subcube['stretch'] = collections.defaultdict(dict)
        subcube['shift'] = collections.defaultdict(dict)
        subcube['raw_counts'] = collections.defaultdict(dict)
        subcube['sky'] = collections.defaultdict(dict)
        subcube['dcentroid'] = collections.defaultdict(dict)
        subcube['dwidth'] = collections.defaultdict(dict)
        subcube['peak'] = collections.defaultdict(dict)

        subcube['centroid'][target] = centroid[target]
        subcube['width'][target] = width[target]
        subcube['stretch'][target] = stretch[target]
        subcube['shift'][target] = shift[target]
        subcube['raw_counts'][target] = raw_counts[target]
        subcube['sky'][target] = sky[target]
        subcube['dcentroid'][target] = dcentroid[target]
        subcube['dwidth'][target] = dwidth[target]
        subcube['peak'][target] = peak[target]
        subcube['wavelengths'][target] = wavelengths[target]
        for i in range(len(comparisons)):
            subcube['centroid'][comparisons[i]] = centroid[comparisons[i]]
            subcube['width'][comparisons[i]] = width[comparisons[i]]
            subcube['stretch'][comparisons[i]] = stretch[comparisons[i]]
            subcube['shift'][comparisons[i]] = shift[comparisons[i]]
            subcube['raw_counts'][comparisons[i]] = raw_counts[comparisons[i]]
            subcube['sky'][comparisons[i]] = sky[comparisons[i]]
            subcube['dcentroid'][comparisons[i]] = dcentroid[comparisons[i]]
            subcube['dwidth'][comparisons[i]] = dwidth[comparisons[i]]
            subcube['peak'][comparisons[i]] = peak[comparisons[i]]
            subcube['wavelengths'][comparisons[i]] = wavelengths[comparisons[i]]
        '''

        self.subcube.append(subcube)

    def makeCompCube(self, subbinindices, n, *binnedok):
        '''A minicube is a subset of a subcube that only includes the relevant wavelength information for a given wavelength bin'''

        self.speak('making compcube for subdirectory number {0}'.format(n))
        #subbinindices are of the format [numexps, numwave]
        self.subbinindices = subbinindices
        self.n = n
        target = self.subcube[self.n]['target']
        comparisons = self.subcube[self.n]['comparisons']

        if binnedok: self.binnedok = binnedok[0]
        else: self.binnedok = np.array([b for b in self.subcube[self.n]['ok']])


        self.compcube = {}
        self.compcube['binnedok'] = self.binnedok
        self.compcube['bjd'] = self.subcube[self.n]['bjd'][self.binnedok]
        self.compcube['norm'] = self.subcube[self.n]['norm'][self.binnedok]

        

        for key in ['airmass', 'rotangle']:
            self.compcube[key] = (self.subcube[self.n][key][self.binnedok] - np.mean(self.subcube[self.n][key][self.binnedok]))/(np.std(self.subcube[self.n][key][self.binnedok]))

        if self.inputs.invvar: 
            self.speak('weighting by inverse variance')
            raw_counts_comps = np.array([np.sum(self.subcube[self.n]['raw_counts'][comparisons[i]] * self.subindices, 1)[self.binnedok] for i in range(len(self.inputs.comparison[self.n]))])
            sky_counts_comps = np.array([np.sum(self.subcube[self.n]['sky'][comparisons[i]] * self.subindices, 1)[self.binnedok] for i in range(len(self.inputs.comparison[self.n]))])
            sig2 = raw_counts_comps + sky_counts_comps
            den = np.sum((1./sig2), 0)

        for key in ['centroid', 'width', 'stretch', 'shift']:
            # detrend against target or comparisons, as specified in inputs
            if self.inputs.against == 'target': keyarray = np.array(self.subcube[self.n][key][target][self.binnedok])
            elif self.inputs.against == 'comparisons': keyarray = np.array([self.subcube[self.n][key][comparisons[i]][self.binnedok] for i in range(len(comparisons))])
            else: self.speak('you have not specified what to detrend against; must be target or comparisons')

            if self.inputs.invvar:
                num = np.sum((keyarray/sig2), 0)
                self.compcube[key] = num/den
            else:
                summed = np.sum(keyarray, 0)
                self.compcube[key] = (summed - np.mean(summed))/np.std(summed)

        for key in ['raw_counts', 'sky', 'dcentroid', 'dwidth', 'peak']:
            # detrend against target or comparisons, as specified in inputs
            if self.inputs.against == 'target': self.keyarray = np.array((self.subcube[self.n][key][target] * self.subbinindices)[self.binnedok])
            elif self.inputs.against == 'comparisons': self.keyarray = np.array([np.sum(self.subcube[self.n][key][comparisons[i]] * self.subbinindices, 1)[self.binnedok] for i in range(len(comparisons))])
            else: self.speak('you have not specified what to detrend against; must be target or comparisons')
            if self.inputs.invvar:
                num = np.sum((keyarray/sig2), 0)
                self.compcube[key] = num/den
            else: 
                summed = np.sum(keyarray, 0)
                self.compcube[key] = (summed - np.mean(summed))/np.std(summed)

        return self.compcube
