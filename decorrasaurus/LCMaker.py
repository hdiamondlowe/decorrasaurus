from .imports import *
from .WaveBinner import WaveBinner
from .Plotter import Plotter

class LCMaker(Talker, Writer):
    '''LCMaker object trims the data in time and also creates light curves for each wavelength bin and saves them in their own .npy files.'''    
    def __init__(self, detrender, subdirectories):
        '''Initialize a LCMaker object.'''

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs.inputs
        self.cube = self.detrender.cube
        self.subcube = self.detrender.cube.subcube
        self.subdirectories = subdirectories

        for n, subdir in enumerate(self.subdirectories):
            self.n = n
            self.subdir = subdir
            self.trimTimeSeries()
            self.maskStarSpot()
        self.makeBinnedLCs()

    def trimTimeSeries(self):
        '''trim the baseline of the time series to  * transit duration on either side of the transit midpoint'''
        # trim the light curves such that there is 1 transit duration on either side of the transit, and not more
        # Tdur needs to be in days
        self.speak('trimming excess baseline by {0} x Tdur for {1}'.format(self.inputs['timesTdur'], self.subdir))
        outside_transit = np.where((self.subcube[self.subdir]['bjd'] < self.inputs[self.subdir]['t0']-(self.inputs['timesTdur']*self.inputs['Tdur'])) | (self.subcube[self.subdir]['bjd'] > self.inputs[self.subdir]['t0']+(self.inputs['timesTdur']*self.inputs['Tdur'])))
        self.subcube[self.subdir]['trimmedok'][outside_transit] = False

    def maskStarSpot(self):

        if self.inputs[self.subdir]['midclip_inds'] == False: pass
        else: 
            self.speak('trimming points from the middle of {0}'.format(self.subdir))
            ind0, ind1 = self.inputs[self.subdir]['midclip_inds'][0], self.inputs[self.subdir]['midclip_inds'][1]
            self.subcube[self.subdir]['trimmedok'][ind0:ind1] = False

    def makeBinnedLCs(self):

        self.wavebin = WaveBinner(self.detrender, self.subdirectories)

        npyfiles = []
        for file in os.listdir(self.detrender.directoryname):
            if file.endswith('.npy'):
                npyfiles.append(file)

        txtfiles = []
        for file in os.listdir(self.detrender.directoryname):
            if file.endswith('.txt'):
                txtfiles.append(file)

        # get a list of all the possible wavebin names from all data sets
        allwavefiles = [self.wavebin.wavefiles[subdir] for subdir in self.subdirectories]
        # sort by ascending order (trickly with strings; we don't want bin 10100-10300 to come before bin 6100-6300
        self.allwavefiles = list(set(np.concatenate(allwavefiles)))
        self.allwavefiles.sort(key = lambda f: float(f.split('-')[0]))

        allwavelims = [self.wavebin.wavelims[subdir] for subdir in self.subdirectories]
        self.allwavelims = sorted(list(set([tuple(wavelim) for wavelim in np.concatenate(allwavelims)])))

        if self.inputs['dividewhite'] and self.inputs['binlen']=='all':
            # save the comparison star white light curves
            self.speak('creating divide white file for later use')
            self.dividewhite = {}

        for w, wavefile in enumerate(self.allwavefiles):

            if wavefile+'.npy' in npyfiles: 
                self.speak(wavefile+' dictionary already exists in the detrender directory')
                pass

            else:
                self.speak('creating dictionary for wavelength bin {0}'.format(wavefile))
                bin = {}
                bin['subdirectories'] = []
                bin['lmfitdone'] = False
                bin['mcfitdone'] = False
                bin['wavelims'] = self.allwavelims[w]

                for n, subdir in enumerate(self.subdirectories):

                    if wavefile in self.wavebin.wavefiles[subdir]:

                        bin['subdirectories'].append(subdir)

                        bininds = self.wavebin.binindices[subdir][wavefile] # shape = (numexps, numstars, numwave)
                        # create a dictionary for this wavelength bin
                        bin[subdir] = {}
                        bin[subdir]['bininds'] = bininds
                        bin[subdir]['binnedok'] = self.subcube[subdir]['mosasaurusok'] * self.subcube[subdir]['trimmedok'] # initially all bad points by wavelength are the same; union of mosasaurusok and trimmed ok points

                        target = self.subcube[subdir]['target']
                        comparisons = self.subcube[subdir]['comparisons']

                        # calculating photon noise limits
                        raw_countsT = np.array(np.sum(self.subcube[subdir]['raw_counts'][target] * bininds, 1))
                        skyT = np.array(np.sum(self.subcube[subdir]['sky'][target] * bininds, 1))
                        raw_countsC = np.sum(np.array([np.sum(self.subcube[subdir]['raw_counts'][comparisons[i]] * bininds, 1) for i in range(len(comparisons))]), 0)
                        skyC = np.sum(np.array([np.sum(self.subcube[subdir]['sky'][comparisons[i]] * bininds, 1) for i in range(len(comparisons))]), 0)
                        sigmaT = np.sqrt(raw_countsT+skyT)/raw_countsT
                        sigmaC = np.sqrt(raw_countsC+skyC)/raw_countsC
                        sigmaF = np.sqrt(sigmaT**2 + sigmaC**2)

                        # initiating writer for this particular wavelength bin output text file
                        if wavefile+'.txt' not in txtfiles: 
                            self.speak('creating output txt file for {0}'.format(wavefile))
                            Writer.__init__(self, self.inputs['directoryname']+wavefile+'.txt')
                            # write a bunch of stuff that you may want easy access too (without loading in the .npy file)
                            self.write('output file for wavelength bin {0}'.format(wavefile))
                            
                            if self.inputs['istarget'] == False: self.write('istarget = false. batman transit model will not be used!')
                            if self.inputs['optext']: self.write('using optimally extracted spectra')
                            if self.inputs['invvar']: self.write('combining comparisons using: inverse variance')
                            else: self.write('combining comparisons using: simple addition')

                        self.write('the following are parameters from subdirectory {0}'.format(subdir))
                        self.write('    target: '+target)
                        self.write('    comparison: '+str(comparisons))

                        self.write('    photon noise limits:')
                        self.write('        target               comparison           T/C')
                        self.write('        '+str(np.mean(sigmaT))+'    '+str(np.mean(sigmaC))+'    '+str(np.mean(sigmaF)))
                        self.write('    fit labels:  '+str(self.inputs[subdir]['fitlabels']))
                        if self.inputs['sysmodel'] == 'GP':
                            self.write('    kernel labels:  '+str(self.inputs[subdir]['kerneltypes']))
                        self.write('    tran labels: '+str(self.inputs[subdir]['tranlabels']))
                        self.write('    tran params: '+str(self.inputs[subdir]['tranparams']))
                        self.write('    tran bounds: '+str(self.inputs[subdir]['tranbounds'][0])+'\n                 '+str(self.inputs[subdir]['tranbounds'][1]))

                        # save the expected photon noise limit for the target/comparisons lightcurve
                        bin[subdir]['photnoiseest'] = sigmaF

                        # make a lightcurves to work off of
                        raw_counts_targ = np.sum(self.subcube[subdir]['raw_counts'][target] * bininds, 1) # shape = (numexps)
                        raw_counts_targ = raw_counts_targ/np.median(raw_counts_targ)
                        raw_counts_comps = np.sum(np.sum([self.subcube[subdir]['raw_counts'][c] * bininds for c in comparisons], 0), 1)
                        raw_counts_comps = raw_counts_comps/np.median(raw_counts_comps)

                        if self.inputs['dividewhite'] and self.inputs['binlen']=='all':
                            # save the comparison star white light curves
                            self.dividewhite[subdir]['compwhitelcs'] = [np.sum(self.subcube[subdir]['raw_counts'][c], 1) for c in comparisons]
                            np.save(self.inputs['directoryname']+'dividewhite.npy', self.dividewhite)

                            bin[subdir]['lc'] = raw_counts_targ/raw_counts_comps
                            bin[subdir]['compcube'] = self.cube.makeCompCube(bininds, subdir)


                        elif self.inputs['dividewhite'] and self.inputs['binlen']!='all':
                            
                            self.dividewhite = np.load(self.inputs['directoryname']+'dividewhite.npy')[()]
                            self.speak('creating Zwhite(t) for bin')
                            Twhite = self.dividewhite['Twhite']
                            Zwhite = np.sum(self.subcube[self.subdir]['raw_counts'][target] * bininds, 1)/Twhite[self.subdir]
                            bin[subdir]['Zwhite'] = Zwhite/np.median(Zwhite)

                            compwhitelcs = np.array(self.dividewhite[subdir]['compwhitelcs'])
                            complcs = np.array([np.sum(self.subcube[subdir]['raw_counts'][c] * bininds, 1) for c in comparisons])
                            Zcomp = np.sum(complcs/compwhitelcs, 0)
                            bin[subdir]['Zcomp'] = Zcomp/np.median(Zcomp)

                            # don't use comparion star for divide white:
                            bin[subdir]['lc'] = raw_counts_targ
                            bin[subdir]['compcube'] = self.cube.makeCompCube(bininds, subdir)

                        else:
                            bin[subdir]['lc'] = raw_counts_targ/raw_counts_comps
                            bin[subdir]['compcube']  = self.cube.makeCompCube(bininds, subdir)

                        # if goind GPs, need to use the information from compcupe to set up the decorrelation parameters

                np.save(self.inputs['directoryname']+wavefile, bin)
                self.speak('saved dictionary for wavelength bin {0}'.format(wavefile))

                plot = Plotter(self.inputs, self.subcube)
                plot.lcplots(bin)




















