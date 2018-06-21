from .imports import *
from .WaveBinner import WaveBinner
from .Plotter import Plotter

class LCMaker(Talker, Writer):
    '''LCMaker object trims the data in time and also creates light curves for each wavelength bin and saves them in their own .npy files.'''    
    def __init__(self, detrender, subdirectories):
        '''Initialize a LCMaker object.'''

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
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
        self.speak('trimming excess baseline by {0} x Tdur for {1}'.format(self.inputs.timesTdur, self.subdir))
        outside_transit = np.where((self.subcube[self.n]['bjd'] < self.inputs.t0[self.n]-(self.inputs.timesTdur*self.inputs.Tdur)) | (self.subcube[self.n]['bjd'] > self.inputs.t0[self.n]+(self.inputs.timesTdur*self.inputs.Tdur)))
        self.subcube[self.n]['trimmedok'][outside_transit] = False

    def maskStarSpot(self):

        if self.inputs.midclip_inds[self.n] == False: pass
        else: 
            self.speak('trimming points from the middle of {0}'.format(self.subdir))
            ind0, ind1 = self.inputs.midclip_inds[self.n][0], self.inputs.midclip_inds[self.n][1]
            self.subcube[self.n]['trimmedok'][ind0:ind1] = False

    def makeBinnedLCs(self):

        self.wavebin = WaveBinner(self.detrender, self.subdirectories)

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

                for n, subdir in enumerate(self.subdirectories):
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
                        bin['binnedok'] = [self.subcube[self.n]['mosasaurusok'] * self.subcube[self.n]['trimmedok']] # initially all bad points by wavelength are the same; union of mosasaurusok and trimmed ok points
                    else:
                        bin['bininds'].append(bininds)
                        bin['binnedok'].append(self.subcube[self.n]['mosasaurusok'] * self.subcube[self.n]['trimmedok'])

                    if n == 0: self.speak('creating output txt file for '+wavefile)

                    target = self.subcube[self.n]['target']
                    comparisons = self.subcube[self.n]['comparisons']

                    # calculating photon noise limits
                    raw_countsT = np.array(np.sum(self.subcube[self.n]['raw_counts'][target] * bininds, 1))
                    skyT = np.array(np.sum(self.subcube[self.n]['sky'][target] * bininds, 1))
                    raw_countsC = np.sum(np.array([np.sum(self.subcube[self.n]['raw_counts'][comparisons[i]] * bininds, 1) for i in range(len(comparisons))]), 0)
                    skyC = np.sum(np.array([np.sum(self.subcube[self.n]['sky'][comparisons[i]] * bininds, 1) for i in range(len(comparisons))]), 0)
                    sigmaT = np.sqrt(raw_countsT+skyT)/raw_countsT
                    sigmaC = np.sqrt(raw_countsC+skyC)/raw_countsC
                    sigmaF = np.sqrt(sigmaT**2 + sigmaC**2)

                    # initiating writer for this particular wavelength bin output text file
                    if self.n == 0: 
                        Writer.__init__(self, self.inputs.saveas+wavefile+'.txt')
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
                    raw_counts_targ = raw_counts_targ/np.median(raw_counts_targ)
                    raw_counts_comps = np.sum(np.sum([self.subcube[self.n]['raw_counts'][c] * bininds for c in comparisons], 0), 1)
                    raw_counts_comps = raw_counts_comps/np.median(raw_counts_comps)

                    # make list of lightcurves and compcubes used for detrending for each night in subdirectories
                    if self.n == 0: 
                        bin['lc'] = [raw_counts_targ/raw_counts_comps]
                        bin['compcube'] = [self.cube.makeCompCube(bininds, self.n)]
                    else: 
                        bin['lc'].append(raw_counts_targ/raw_counts_comps)
                        bin['compcube'].append(self.cube.makeCompCube(bininds, self.n))

                    if self.inputs.dividewhite and self.inputs.binlen!='all':
                        
                        self.dividewhite = np.load(self.inputs.saveas+'dividewhite.npy')[()]
                        self.speak('creating Zwhite(t) for bin')
                        Twhite = self.dividewhite['Twhite']
                        Zwhite = np.sum(self.subcube[self.n]['raw_counts'][target] * bininds, 1)/Twhite[self.n]
                        if self.n == 0: bin['Zwhite'] = Zwhite/np.median(Zwhite)
                        else: bin['Zwhite'].append(Zwhite/np.median(Zwhite))

                        compwhitelcs = self.dividewhite['compwhitelcs'][self.n]
                        #Zcomp = 

                    if self.inputs.dividewhite and self.inputs.binlen=='all':
                        # save the comparison star white light curves
                        self.speak('creating divide white file for later use')
                        self.dividewhite = {}
                        if self.n == 0: self.dividewhite['compwhitelcs'] = [np.sum(self.subcube[self.n]['raw_counts'][c], 1) for c in comparisons]
                        else: self.dividewhite['compwhitelcs'].append([np.sum(self.subcube[self.n]['raw_counts'][c], 1) for c in comparisons])
                        np.save(self.inputs.saveas+'dividewhite.npy', self.dividewhite)

                np.save(self.inputs.saveas+wavefile, bin)
                self.speak('saved dictionary for wavelength bin {0}'.format(wavefile))

                plot = Plotter(self.inputs, self.subcube)
                plot.lcplots(bin)


