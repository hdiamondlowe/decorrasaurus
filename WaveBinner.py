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
            if 'wavebin' in self.subcube[self.n].keys():
                self.speak('reading in wavebin parameters from subcube saved in {0}'.format(self.detrender.directoryname))
                self.binindices.append(self.subcube[self.n]['wavebin']['binindices'])
                if self.n == 0:
                    self.wavelims = self.subcube[self.n]['wavebin']['wavelims']
                    self.wavefiles = self.subcube[self.n]['wavebin']['wavefiles']
            else: 
                self.makeBinIndices()
                self.speak('saving wavebin properties to the subcube')
                np.save(self.inputs.saveas+'_subcube.npy', self.subcube)


    def makeBinIndices(self):

        self.speak('creating binindices array for {0} which will help to make wavelength binned lighcurves'.format(self.subdir))

        numexps, numwave = len(self.subcube[self.n]['bjd']), len(self.subcube[self.n]['wavelengths'])

        if self.n == 0:
            waverange = self.inputs.wavelength_lims[1] - self.inputs.wavelength_lims[0]
            if self.inputs.binlen == 'all': self.binlen = waverange
            else: self.binlen = self.inputs.binlen
            self.numbins = int(np.floor(waverange/self.binlen))
            self.binlen = waverange/float(self.numbins)
            self.wavelims = []
            [self.wavelims.append((self.inputs.wavelength_lims[0]+(i*self.binlen), self.inputs.wavelength_lims[0]+((i+1)*self.binlen))) for i in range(int(self.numbins))]
        binindices = np.zeros((self.numbins, numwave))

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
            

            #minwaveind = np.where(self.subcube[self.n]['wavelengths'] == np.ceil(wavelim[0]))[0][0]
            #print(minwaveind)
            #minwaveextra = minwaveind - wavelim[0]
            #maxwaveind = int(np.floor(wavelim[1]))
            #maxwaveextra = wavelim[1] - maxwaveind
            #indarray = np.zeros(numwave)
            #indarray[minwaveind:maxwaveind] = 1
            #indarray[minwaveind-1] = minwaveextra
            #indarray[maxwaveind] = maxwaveextra

            #binindices[n][0][i] = indarray

        
        '''
        #self.binnedcube_targ = np.zeros((numexps, self.numbins, numwave))
        for n in range(numexps):
            for i, wavelim in enumerate(self.wavelims):
                  
                minwave_interp1 = np.interp(wavelim[0], self.subcube[self.n]['wavelengths'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]][n], starmaster['wavelength'])
                minwave_interp = np.interp(minwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                maxwave_interp1 = np.interp(wavelim[1], self.subcube[self.n]['wavelengths'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]][n], starmaster['wavelength'])
                maxwave_interp = np.interp(maxwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                minwaveind = int(np.ceil(minwave_interp))
                minwaveextra = minwaveind - minwave_interp
                maxwaveind = int(np.floor(maxwave_interp))
                maxwaveextra = maxwave_interp - maxwaveind
                indarray = np.zeros((numwave))
                indarray[minwaveind:maxwaveind] = 1
                indarray[minwaveind-1] = minwaveextra
                indarray[maxwaveind] = maxwaveextra

                binindices[n][0][i] = indarray

        for n in range(numexps):
            for s in range(len(self.inputs.comparison[self.n])):
                for i, wavelim in enumerate(self.wavelims):
                      
                    minwave_interp1 = np.interp(wavelim[0], self.subcube[self.n]['wavelengths'][self.inputs.comparison[self.n][s]][self.inputs.comparisonpx[self.n][s]][n], starmaster['wavelength'])
                    minwave_interp = np.interp(minwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                    maxwave_interp1 = np.interp(wavelim[1], self.subcube[self.n]['wavelengths'][self.inputs.comparison[self.n][s]][self.inputs.comparisonpx[self.n][s]][n], starmaster['wavelength'])
                    maxwave_interp = np.interp(maxwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                    minwaveind = int(np.ceil(minwave_interp))
                    minwaveextra = minwaveind - minwave_interp
                    maxwaveind = int(np.floor(maxwave_interp))
                    maxwaveextra = maxwave_interp - maxwaveind
                    indarray = np.zeros((numwave))
                    indarray[minwaveind:maxwaveind] = 1
                    indarray[minwaveind-1] = minwaveextra
                    indarray[maxwaveind] = maxwaveextra

                    binindices[n][s+1][i] = indarray
        '''
        self.binindices.append(binindices)

        self.subcube[self.n]['wavebin'] = {}
        self.subcube[self.n]['wavebin']['binindices'] = binindices
        self.subcube[self.n]['wavebin']['wavelims'] = self.wavelims
        if self.n == 0: self.wavefiles = [str(i[0])+'-'+str(i[1]) for i in self.wavelims]
        self.subcube[self.n]['wavebin']['wavefiles'] = self.wavefiles


