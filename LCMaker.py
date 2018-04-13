from .imports import *
from .WaveBinner import WaveBinner

class LCMaker(Talker, Writer):
    '''LCMaker object trims the data in time and also creates light curves for each wavelength bin and saves them in their own .npy files.'''    
    def __init__(self, detrender, directories):
        '''Initialize a LCMaker object.'''

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.cube = self.detrender.cube
        self.subcube = self.detrender.cube.subcube
        self.directories = directories

        for n, subdir in enumerate(self.directories):
            self.n = n
            self.subdir = subdir
            self.trimTimeSeries()
        self.makeBinnedLCs()

    def trimTimeSeries(self):
        '''trim the baseline of the time series to 1.5 * transit duration on either side of the transit midpoint'''
        # trim the light curves such that there is 1 transit duration on either side of the transit, and not more
        # Tdur needs to be in days
        self.speak('trimming excess baseline from {0}'.format(self.subdir))
        outside_transit = np.where((self.subcube[self.n]['bjd'] < self.inputs.t0[self.n]-(1.5*self.inputs.Tdur)) | (self.subcube[self.n]['bjd'] > self.inputs.t0[self.n]+(1.5*self.inputs.Tdur)))
        # !!! may have issue here when changing the midpoint time; may need to reset self.ok to all True and then add in time clip
        self.subcube[self.n]['ok'][outside_transit] = False

    def makeBinnedLCs(self):

        self.wavebin = WaveBinner(self.detrender, self.directories)

        npyfiles = []
        for file in os.listdir(self.detrender.directoryname):
            if file.endswith('.npy'):
                npyfiles.append(file)

        for w, wavefile in enumerate(self.wavebin.wavefiles):

            #file = self.inputs.nightname+'_'+str(wavelims[0])+'-'+str(wavelims[1])+'.npy'
            if wavefile+'.npy' in npyfiles: 
                self.speak(wavefile+' dictionary already exists in the detrender directory')
                pass

            else:

                for n, subdir in enumerate(self.directories):
                    self.n = n
                    self.subdir = subdir

                    #basename = os.path.splitext(wavefile)[0][14:]
                    if self.n == 0: self.speak('creating dictionary for wavelength bin {0}'.format(wavefile))

                    bininds = self.wavebin.binindices[self.n][w] # shape = (numexps, numstars, numwave)
                    # create a dictionary for this wavelength bin
                    if self.n == 0:
                        bin = {}
                        bin['freeparams'] = self.inputs.freeparamnames
                        bin['wavelims'] = self.wavebin.wavelims[w]
                        bin['bininds'] = [bininds]
                        bin['binnedok'] = [np.array([b for b in self.subcube[self.n]['ok']])]
                    else:
                        bin['bininds'].append(bininds)
                        bin['binnedok'].append(np.array([b for b in self.subcube[self.n]['ok']]))

                    if n == 0: self.speak('creating output txt file for '+wavefile)

                    target = self.subcube[self.n]['target']
                    comparisons = self.subcube[self.n]['comparisons']

                    # calculating photon noise limits
                    raw_countsT = np.array(np.sum(self.subcube[self.n]['raw_counts'][target] * bininds, 1)[self.subcube[self.n]['ok']])
                    skyT = np.array(np.sum(self.subcube[self.n]['sky'][target] * bininds, 1)[self.subcube[self.n]['ok']])
                    raw_countsC = np.sum(np.array([np.sum(self.subcube[self.n]['raw_counts'][comparisons[i]] * bininds, 1)[self.subcube[self.n]['ok']] for i in range(len(comparisons))]), 0)
                    skyC = np.sum(np.array([np.sum(self.subcube[self.n]['sky'][comparisons[i]] * bininds, 1)[self.subcube[self.n]['ok']] for i in range(len(comparisons))]), 0)
                    sigmaT = np.sqrt(raw_countsT+skyT)/raw_countsT
                    sigmaC = np.sqrt(raw_countsC+skyC)/raw_countsC
                    sigmaF = np.sqrt(sigmaT**2 + sigmaC**2)

                    # initiating writer for this particular wavelength bin output text file
                    if self.n == 0: 
                        Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')
                        # write a bunch of stuff that you may want easy access too (without loading in the .npy file)
                        self.write('output file for wavelength bin '+wavefile)
                        
                        if self.inputs.istarget == False: self.write('istarget = false. batman transit model will not be used!')
                        if self.inputs.optext: self.write('using optimally extracted spectra')
                        if self.inputs.invvar: self.write('combining comparisons using: inverse variance')
                        else: self.write('combining comparisons using: simple addition')

                    self.write('the following are parameters from subdirectory {0}'.format(self.subdir))
                    self.write('    target: '+target)
                    self.write('    comparison: '+str(comparisons))

                    self.write('    photon noise limits:')
                    self.write('        target               comparison           T/C')
                    self.write('        '+str(np.mean(sigmaT))+'    '+str(np.mean(sigmaC))+'    '+str(np.mean(sigmaF)))
                    self.write('    fit labels: '+str(self.inputs.fitlabels[self.n]))
                    self.write('    tran labels: '+str(self.inputs.tranlabels[self.n]))
                    self.write('    tran params: '+str(self.inputs.tranparams[self.n]))
                    self.write('    tran bounds: '+str(self.inputs.tranbounds[self.n][0])+'\n                 '+str(self.inputs.tranbounds[self.n][1]))

                    # save the expected photon noise limit for the target/comparisons lightcurve
                    if self.n == 0: bin['photnoiseest'] = [sigmaF]
                    else: bin['photnoiseest'].append(sigmaF)

                    # make a lightcurves to work off of
                    raw_counts_targ = np.sum(self.subcube[self.n]['raw_counts'][target] * bininds, 1) # shape = (numexps)
                    raw_counts_targ = raw_counts_targ/np.mean(raw_counts_targ)
                    raw_counts_comps = np.sum(np.sum([self.subcube[self.n]['raw_counts'][comparisons[i]] * bininds for i in range(len(comparisons))], 0), 1)
                    raw_counts_comps = raw_counts_comps/np.mean(raw_counts_comps)

                    # make list of lightcurves and compcubes used for detrending for each night in directories
                    if self.n == 0: 
                        bin['lc'] = [(raw_counts_targ/np.mean(raw_counts_targ))[self.subcube[self.n]['ok']]/(raw_counts_comps/np.mean(raw_counts_comps))[self.subcube[self.n]['ok']]]
                        bin['compcube'] = [self.cube.makeCompCube(bininds, self.n)]
                    else: 
                        bin['lc'].append((raw_counts_targ/np.mean(raw_counts_targ))[self.subcube[self.n]['ok']]/(raw_counts_comps/np.mean(raw_counts_comps))[self.subcube[self.n]['ok']])
                        bin['compcube'].append(self.cube.makeCompCube(bininds, self.n))

                np.save(self.inputs.saveas+'_'+wavefile, bin)
                self.speak('saved dictionary for wavelength bin {0}'.format(wavefile))

