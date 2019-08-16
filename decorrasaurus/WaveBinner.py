from .imports import *

class WaveBinner(Talker):

    '''this class will bin the data into even wavelength bins'''

    def __init__(self, detrender, directories):

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs.inputs
        self.subcube = self.detrender.cube.subcube
        self.directories = directories

        self.binindices = {}
        self.wavelims = {}
        self.wavefiles = {}

        for subdir in self.directories:
            self.subdir = subdir
            if self.inputs['dividewhite']:
                # if divideWhite want to make new wavelims when switching from binlen='all' to binlen='200'
                self.speak('making wavebins for divide white')
                self.makeBinIndices()
                self.speak('saving wavebin properties to the subcube')
                np.save(self.inputs['directoryname']+'subcube.npy', self.subcube)
            else: 
                self.makeBinIndices()
                #if self.inputs['fullsample']:
                self.speak('saving wavefiles to subcube')
                np.save(self.inputs['directoryname']+'subcube.npy', self.subcube)


    def makeBinIndices(self):

        self.speak('creating binindices array for {0} which will help to make wavelength binned lighcurves'.format(self.subdir))

        numexps, numwave = len(self.subcube[self.subdir]['bjd']), len(self.subcube[self.subdir]['wavelengths'])

        waverange = self.inputs[self.subdir]['wavelength_lims'][1] - self.inputs[self.subdir]['wavelength_lims'][0]
        if self.inputs['binlen'] == 'all': self.binlen = waverange
        else: self.binlen = self.inputs['binlen']
        self.numbins = int(np.floor(waverange/self.binlen))
        self.binlen = waverange/float(self.numbins)
        self.wavelims[self.subdir] = [(self.inputs[self.subdir]['wavelength_lims'][0]+(i*self.binlen), self.inputs[self.subdir]['wavelength_lims'][0]+((i+1)*self.binlen)) for i in range(int(self.numbins))]
        binindices = {}

        for i, wavelim in enumerate(self.wavelims[self.subdir]):

            wavefile = str(wavelim[0])+'-'+str(wavelim[1])

            # make an array that will be a mask for the wavelength parameter; array of indices
            indarray = np.zeros(numwave)

            # round up minimum wavelim to nearest integer
            minwaveroundup = int(np.ceil(wavelim[0]))
            # find how much extra is needed in wavelength spave
            minwaveextra = minwaveroundup - wavelim[0]
            # find what wavelength integer corresponds to these
            minwaveind = np.where(self.subcube[self.subdir]['wavelengths'] == minwaveroundup)[0][0]

            # do the same for the maximum wavelim
            maxwaverounddown = int(np.floor(wavelim[1]))
            maxwaveextra = wavelim[1] - maxwaverounddown

            maxwaveind = np.where(self.subcube[self.subdir]['wavelengths'] == maxwaverounddown)[0][0]

            # set the hard wavelength bounds to 1
            indarray[minwaveind:maxwaveind] = 1
            # set the edges to the fractions to be included
            indarray[minwaveind-1] = minwaveextra
            indarray[maxwaveind] = maxwaveextra

            # put this in the master binindices so that it can be used later for binning
            binindices[wavefile] = indarray
            
        self.binindices[self.subdir] = binindices
        self.wavefiles[self.subdir] = [str(i[0])+'-'+str(i[1]) for i in self.wavelims[self.subdir]]
        self.subcube[self.subdir]['wavefiles'] = self.wavefiles[self.subdir]
      

