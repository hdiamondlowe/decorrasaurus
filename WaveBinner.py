from .imports import *

class WaveBinner(Talker):

    '''this class will bin the data into even wavelength bins'''

    def __init__(self, detrender, directories):

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.subcube = self.detrender.cube.subcube
        self.directories = directories

        self.binindices = []

        for n, subdir in enumerate(self.directories):
            self.n = n
            self.subdir = subdir
            if self.inputs.dividewhite:
                # if divideWhite want to make new wavelims when switching from binlen='all' to binlen='200'
                self.speak('making wavebins for divide white')
                self.makeBinIndices()
                self.speak('saving wavebin properties to the subcube')
                np.save(self.inputs.saveas+'subcube.npy', self.subcube)
            elif 'wavebin' in self.subcube[self.n].keys():
                self.speak('reading in wavebin parameters from subcube saved in {0}'.format(self.detrender.directoryname))
                self.binindices.append(self.subcube[self.n]['wavebin']['binindices'])
                if self.n == 0:
                    self.wavelims = self.subcube[self.n]['wavebin']['wavelims']
                    self.wavefiles = self.subcube[self.n]['wavebin']['wavefiles']
            else: 
                self.makeBinIndices()
                self.speak('saving wavebin properties to the subcube')
                np.save(self.inputs.saveas+'subcube.npy', self.subcube)


    def makeBinIndices(self):

        self.speak('creating binindices array for {0} which will help to make wavelength binned lighcurves'.format(self.subdir))

        numexps, numwave = len(self.subcube[self.n]['bjd']), len(self.subcube[self.n]['wavelengths'])

        if self.n == 0:
            self.wavelims = []

        waverange = self.inputs.wavelength_lims[self.n][1] - self.inputs.wavelength_lims[self.n][0]
        if self.inputs.binlen == 'all': self.binlen = waverange
        else: self.binlen = self.inputs.binlen
        self.numbins = int(np.floor(waverange/self.binlen))
        self.binlen = waverange/float(self.numbins)
        [self.wavelims.append([(self.inputs.wavelength_lims[self.n][0]+(i*self.binlen), self.inputs.wavelength_lims[self.n][0]+((i+1)*self.binlen))]) for i in range(int(self.numbins))]
        binindices = np.zeros((self.numbins, numwave))
        print(self.wavelims)
        '''
        for i, wavelim in enumerate(self.wavelims):
            # make an array that will be a mask for the wavelength parameter; array of indices
            indarray = np.zeros(numwave)

            # round up minimum wavelim to nearest integer
            minwaveroundup = int(np.ceil(wavelim[0]))
            # find how much extra is needed in wavelength spave
            minwaveextra = minwaveroundup - wavelim[0]
            # find what wavelength integer corresponds to these
            minwaveind = np.where(self.subcube[self.n]['wavelengths'] == minwaveroundup)[0][0]

            # do the same for the maximum wavelim
            maxwaverounddown = int(np.floor(wavelim[1]))
            maxwaveextra = wavelim[1] - maxwaverounddown

            maxwaveind = np.where(self.subcube[self.n]['wavelengths'] == maxwaverounddown)[0][0]

            # set the hard wavelength bounds to 1
            indarray[minwaveind:maxwaveind] = 1
            # set the edges to the fractions to be included
            indarray[minwaveind-1] = minwaveextra
            indarray[maxwaveind] = maxwaveextra

            # put this in the master binindices so that it can be used later for binning
            binindices[i] = indarray
            
        self.binindices.append(binindices)

        self.subcube[self.n]['wavebin'] = {}
        self.subcube[self.n]['wavebin']['binindices'] = binindices
        self.subcube[self.n]['wavebin']['wavelims'] = self.wavelims
        if self.n == 0: self.wavefiles = [str(i[0])+'-'+str(i[1]) for i in self.wavelims]
        self.subcube[self.n]['wavebin']['wavefiles'] = self.wavefiles
        '''

