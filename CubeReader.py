from .imports import *
from .Plotter import Plotter
from copy import deepcopy

class CubeReader(Talker):
    ''' Reads in all the datacubes from the local subdirectories'''
    def __init__(self, detrender, subdirectories):

        Talker.__init__(self)
        
        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.subdirectories = subdirectories

        try: 
            self.speak('trying to read in subcube')
            self.subcube = np.load(self.inputs.saveas+'subcube.npy')[()]
            self.speak('loaded in subcube') 
        except(IOError):
            self.speak('subcube does not exist, creating a new one')
            self.subcube = [] # make a list of subcubes for all nights involved
            for n, subdir in enumerate(self.subdirectories):
                self.n = n
                self.subdir = subdir
                self.datacubepath = self.inputs.datacubepath[self.n]
                self.specstretchpath = self.inputs.specstretchpath[self.n]
                self.makeSubCube()
            np.save(self.inputs.saveas+'subcube.npy', self.subcube)
            self.speak('subcube saved')

            if self.inputs.makeplots:
                plot = Plotter(self.inputs, self.subcube)
                plot.cubeplots()

    def makeSubCube(self):

        self.speak('reading in datacube from {0} and extracting the arrays you care about'.format(self.subdir))
        cube = np.load(self.datacubepath)[()]
        spec = np.load(self.specstretchpath)[()]

        self.speak('making subcube of just the arrays you care about from the full cube from {0}'.format(self.subdir))
        
        subcube = {}

        subcube['target'] = cube['target']
        subcube['comparisons'] = cube['comparisons']
        #subcube['target'] = cube['comparisons'][1]
        #subcube['comparisons'] = [cube['comparisons'][3]]

        subcube['mosasaurusok'] = deepcopy(cube['temporal']['ok'])     # (time)
        subcube['trimmedok'] = np.ones_like(subcube['mosasaurusok'])  # (time) making this ahead of time for when we trim the light curve in time (LCMaker)
        subcube['bjd'] = deepcopy(cube['temporal']['bjd'])             # (time)
        subcube['airmass'] = deepcopy(cube['temporal']['airmass'])     # (time)
        subcube['rotangle'] = deepcopy(cube['temporal']['rotatore'])   # (time)

        subcube['norm'] = np.ones(len(subcube['bjd']))                            # normalization constant (time)

        subcube['wavelengths'] = cube['spectral']['wavelength']        # (wave)

        # have to re-make these dictionaries
        subcube['centroid'] = deepcopy(cube['squares']['centroid'])    # [star](time)
        subcube['width'] = deepcopy(cube['squares']['width'])          # [star](time)
        #subcube['median_width'] = deepcopy(cube['squares']['median_width']) # [star](time)
        subcube['stretch'] = deepcopy(spec['stretch'])                 # [star](time)
        subcube['shift'] = deepcopy(spec['shift'])          # [star](time)

        subcube['raw_counts'] = deepcopy(cube['cubes']['raw_counts'])  # [star](time, wave)
        subcube['sky'] = deepcopy(cube['cubes']['sky'])                # [star](time, wave)
        subcube['dcentroid'] = deepcopy(cube['cubes']['centroid'])     # [star](time, wave)
        subcube['dwidth'] = deepcopy(cube['cubes']['width'])           # [star](time, wave)
        subcube['peak'] = deepcopy(cube['cubes']['peak'])              # [star](time, wave)

        self.subcube.append(subcube)

    def makeCompCube(self, subbinindices, n):
        '''A minicube is a subset of a subcube that only includes the relevant wavelength information for a given wavelength bin'''

        self.speak('making compcube for subdirectory number {0}'.format(n))
        #subbinindices are of the format [numexps, numwave]
        self.subbinindices = subbinindices
        self.n = n
        target = self.subcube[self.n]['target']
        comparisons = self.subcube[self.n]['comparisons']

        #if binnedok: self.binnedok = binnedok[0]
        #else: self.binnedok = np.array([b for b in self.subcube[self.n]['mosasaurusok']])


        self.compcube = {}
        #self.compcube['binnedok'] = self.binnedok
        self.compcube['bjd'] = self.subcube[self.n]['bjd']
        self.compcube['norm'] = self.subcube[self.n]['norm']

        for key in self.inputs.fitlabels[self.n]:

            if key in ['airmass', 'rotangle']:
                self.compcube[key] = (self.subcube[self.n][key] - np.mean(self.subcube[self.n][key]))/(np.std(self.subcube[self.n][key]))

            if self.inputs.invvar: 
                self.speak('weighting by inverse variance')
                raw_counts_comps = np.array([np.sum(self.subcube[self.n]['raw_counts'][comparisons[i]] * self.subindices, 1) for i in range(len(self.inputs.comparison[self.n]))])
                sky_counts_comps = np.array([np.sum(self.subcube[self.n]['sky'][comparisons[i]] * self.subindices, 1) for i in range(len(self.inputs.comparison[self.n]))])
                sig2 = raw_counts_comps + sky_counts_comps
                den = np.sum((1./sig2), 0)

            if key in ['centroid', 'width', 'median_width']:#, 'stretch', 'shift']:
                # detrend against target or comparisons, as specified in inputs
                if self.inputs.against == 'target': keyarray = np.array([self.subcube[self.n][key][target]])
                elif self.inputs.against == 'comparisons': keyarray = np.array([self.subcube[self.n][key][comparisons[i]] for i in range(len(comparisons))])
                elif self.inputs.against == 'difference': keyarray = np.array([self.subcube[self.n][key][target]]) - np.array([self.subcube[self.n][key][comparisons[i]] for i in range(len(comparisons))])
                else: self.speak('you have not specified what to detrend against; must be target or comparisons')

                if self.inputs.invvar:
                    num = np.sum((keyarray/sig2), 0)
                    self.compcube[key] = num/den
                else:
                    summed = np.sum(keyarray, 0)
                    self.compcube[key] = (summed - np.mean(summed))/np.std(summed)

            if key in ['stretch', 'shift']:
                # detrend against target or comparisons, as specified in inputs
                if self.inputs.against == 'target': keyarray = np.array([list(self.subcube[self.n][key][target].values())])
                elif self.inputs.against == 'comparisons': keyarray = np.array([list(self.subcube[self.n][key][comparisons[i]].values()) for i in range(len(comparisons))])
                elif self.inputs.against == 'difference': keyarray = np.array([list(self.subcube[self.n][key][target].values())]) - np.array([list(self.subcube[self.n][key][comparisons[i]].values()) for i in range(len(comparisons))])
                else: self.speak('you have not specified what to detrend against; must be target or comparisons')

                if self.inputs.invvar:
                    num = np.sum((keyarray/sig2), 0)
                    self.compcube[key] = num/den
                else:
                    summed = np.sum(keyarray, 0)
                    self.compcube[key] = (summed - np.mean(summed))/np.std(summed)

            if key in ['raw_counts', 'sky', 'dcentroid', 'dwidth', 'peak']:
                # detrend against target or comparisons, as specified in inputs
                if self.inputs.against == 'target': keyarray = np.array([np.nansum(self.subcube[self.n][key][target] * self.subbinindices, 1)])
                elif self.inputs.against == 'comparisons': keyarray = np.array([np.nansum(self.subcube[self.n][key][comparisons[i]] * self.subbinindices, 1) for i in range(len(comparisons))])
                elif self.inputs.against == 'difference': keyarray = np.array([np.nansum(self.subcube[self.n][key][target] * self.subbinindices, 1)]) - np.array([np.nansum(self.subcube[self.n][key][comparisons[i]] * self.subbinindices, 1) for i in range(len(comparisons))])
                else: self.speak('you have not specified what to detrend against; must be target or comparisons')
                if self.inputs.invvar:
                    num = np.sum((keyarray/sig2), 0)
                    self.compcube[key] = num/den
                else: 
                    summed = np.sum(keyarray, 0)
                    self.compcube[key] = (summed - np.mean(summed))/np.std(summed)
                
        return self.compcube
