from imports import *
from shutil import copyfile
from datetime import datetime
from astropy.table import Table
import string

#  an object that reads in an input.init file, stores all of the information, and also makes a copy of that file to the directory everything will get saved to
class Inputs(Talker):
    '''InputsJoint object reads input.init information from all subdirectories and copies those files to the working detrender directry (if it's not already there).'''
    def __init__(self, subdirectories, *directoryname):
        '''Initialize an Inputs object.'''

        Talker.__init__(self)

        self.inputs = {}
        self.inputs['subdirectories'] = subdirectories


        if not directoryname: self.createDirectory()
        else: self.directoryname = directoryname[0]

        for n, subdir in enumerate(self.inputs['subdirectories']):
            self.n = n
            self.subdir = subdir
            self.speak('going into directory {0}'.format(self.directoryname))
            self.readInputs()

        self.speak('successfully collected input parameters')

    def createDirectory(self):
        ''' Create a new directory to put detrender stuff in'''

        subdir = self.inputs['subdirectories'][0]

        self.speak('creating directory from input file {0}'.format(subdir))

        file = open(subdir+'/input.init')
        lines = file.readlines()
        dictionary = {}
        for i in range(3):
          if lines[i] != '\n' and lines[i][0] != '#':
            split = lines[i].split()
            key = split[0]
            entries = split[1:]
            if len(entries) == 1:
              entries = entries[0]
            dictionary[key] = entries


        #if self.n == 0:
        self.speak('creating new detrender directory')
        self.speak('where possible, input values from {0} will be used'.format(subdir))
        self.filename = dictionary['filename']


        # create working folder for the files
        dt = datetime.now()
        runpath = dt.strftime('%Y-%m-%d-%H:%M_')+self.filename+'/'
        directorypath = 'run' + '/'
        if not os.path.exists(directorypath):
            os.makedirs(directorypath)
        if os.path.exists(directorypath+runpath):
            self.speak('run path already exists! you are writing over some files...')
        else:
            os.makedirs(directorypath+runpath)

        self.directoryname = directorypath+runpath

        for subdir in self.inputs['subdirectories']:
            self.speak('copying {0} to directory {1}'.format(subdir+'/input.init', self.directoryname))
            copyfile(subdir+'/input.init', self.directoryname+subdir+'_input.init')
            
    def readInputs(self):

        self.speak('reading {0} input file from {1}'.format(self.subdir, self.directoryname))

        try: file = open(self.directoryname+self.subdir+'_input.init')
        except(FileNotFoundError): 
            print('The input file you are looking for does not seem to exist. Inputs will not be collected.')
            return
        lines = file.readlines()
        dictionary = {}
        for i in range(len(lines)):
          if lines[i] != '\n' and lines[i][0] != '#':
            split = lines[i].split()
            key = split[0]
            entries = split[1:]
            if len(entries) == 1:
              entries = entries[0]
            dictionary[key] = entries

        def str_to_bool(s):
            if s == 'True':
                return True
            elif s == 'False':
                return False
            elif s == 'None':
                return None
            else:
                try: return float(s)
                except(ValueError): 
                    if s in dictionary.keys(): return float(dictionary[s])
                    else: return str(s)

        inputs = {}
        inputs['n'] = str(self.n)
        inputs['nightname']  = dictionary['nightname']

        if self.n == 0:

            self.inputs['directoryname'] = self.directoryname

            self.inputs['Teff']      = float(dictionary['Teff'])
            self.inputs['Teff_unc']  = float(dictionary['Teff_unc'])
            self.inputs['logg']      = float(dictionary['logg'])
            self.inputs['logg_unc']  = float(dictionary['logg_unc'])
            self.inputs['z']         = float(dictionary['z'])
            self.inputs['z_unc']     = float(dictionary['z_unc'])
            self.inputs['ldlaw']     = str_to_bool(dictionary['ldlaw'])

            self.inputs['T0']        = float(dictionary['T0'])
            self.inputs['P']         = float(dictionary['P'])
            self.inputs['Tdur']      = float(dictionary['Tdur'])
            self.inputs['inc']       = float(dictionary['inc'])
            self.inputs['a']         = float(dictionary['a'])
            self.inputs['ecc']       = float(dictionary['ecc'])
            self.inputs['omega']     = float(dictionary['omega'])

            self.inputs['modelcode'] = dictionary['modelcode']
 
        inputs['epochnum'] = int(dictionary['epochnum'])
        inputs['toff'] = self.inputs['T0'] + self.inputs['P']*inputs['epochnum']

        if self.n == 0:
            self.inputs['sysmodel'] = dictionary['sysmodel']

        inputs['fitlabels']  = dictionary['fitlabels']
        if type(inputs['fitlabels']) == str: inputs['fitlabels'] = [dictionary['fitlabels']]

        if self.inputs['sysmodel'] == 'linear':
            inputs['polyfit']    = int(dictionary['polyfit'])
            inputs['polylabels'] = ['P{0}coeff'.format(x) for x in range(inputs['polyfit'])]
            inputs['fitparams']  = [1 for f in inputs['fitlabels']]
            inputs['polyparams'] = [1 for p in inputs['polylabels']]

        elif self.inputs['sysmodel'] == 'GP':
            inputs['kerneltypes']  = dictionary['kerneltypes']
            if type(inputs['kerneltypes']) == str: inputs['kerneltypes'] = [dictionary['kerneltypes']]
            assert len(inputs['fitlabels']) == len(inputs['kerneltypes']), 'You need the same number of kernels are there are decorrelation parameters'
            # later the boundary and starting values will be calculated from the fit parameters chosen
            listofGeorgekernels = ['Matern52Kernel', 'ExpSquaredKernel', 'Matern32Kernel', 'RationalQuadraticKernel', 'ExpKernel']
            for kernelname in inputs['kerneltypes']:
                assert kernelname in listofGeorgekernels, 'You need to put in valid kernel types; check the George documentation if you have to'

        else: 
            self.speak('ERROR! Need to define a decorrelation model for the systematics, either linear or GP')
            return

        inputs['tranlabels']      = dictionary['tranlabels']
        if self.inputs['ldlaw'] in ['qd', 'sq', 'lg']: assert('u3' not in inputs['tranlabels']), 'Check limb darkening law with number of ld parameters'
        elif self.inputs['ldlaw'] == 'nl': assert('u3' in inputs['tranlabels']), 'Check limb darkening law with number of ld parameters'

        inputs['tranparams']      = [str_to_bool(i) for i in dictionary['tranparams']]
        inputs['tranbounds']      = [[str_to_bool(i) for i in dictionary['tranbounds_low']], [str_to_bool(i) for i in dictionary['tranbounds_high']]]
        inputs['wavelength_lims'] = [float(i) for i in dictionary['wavelength_lims']]
        assert(len(inputs['tranbounds'][0]) == len(inputs['tranlabels']) and len(inputs['tranbounds'][1]) == len(inputs['tranlabels'])), 'There is something wrong with the transit parameter bounds'
        inputs['against']         = dictionary['against']

        inputs['freeparambounds'] = [[], []]
        inputs['freeparamnames'] = []
        inputs['freeparamvalues'] = []

        if self.inputs['sysmodel'] == 'linear':

            for p, plabel in enumerate(inputs['polylabels']):
                inputs['freeparambounds'][0].append(True)
                inputs['freeparambounds'][1].append(True)
                inputs['freeparamnames'].append(plabel+str(self.n))
                inputs['freeparamvalues'].append(inputs['polyparams'][p])

            for f, flabel in enumerate(inputs['fitlabels']):
                inputs['freeparambounds'][0].append(True)
                inputs['freeparambounds'][1].append(True)
                inputs['freeparamnames'].append(flabel+str(self.n))
                inputs['freeparamvalues'].append(inputs['fitparams'][f])

            for t, tlabel in enumerate(inputs['tranlabels']):
                if type(inputs['tranbounds'][0][t]) == bool and inputs['tranbounds'][0][t] == False: continue
                inputs['freeparambounds'][0].append(inputs['tranbounds'][0][t])
                inputs['freeparambounds'][1].append(inputs['tranbounds'][1][t])
                inputs['freeparamnames'].append(tlabel+str(self.n))
                inputs['freeparamvalues'].append(inputs['tranparams'][t])

        elif self.inputs['sysmodel'] == 'GP':

            for t, tlabel in enumerate(inputs['tranlabels']):
                if type(inputs['tranbounds'][0][t]) == bool and inputs['tranbounds'][0][t] == False: continue
                if inputs['tranbounds'][0][t] == True: inputs['freeparambounds'][0].append(None)
                else: inputs['freeparambounds'][0].append(inputs['tranbounds'][0][t])
                if inputs['tranbounds'][1][t] == True: inputs['freeparambounds'][1].append(None)
                else: inputs['freeparambounds'][1].append(inputs['tranbounds'][1][t])
                inputs['freeparamnames'].append(tlabel+str(self.n))
                inputs['freeparamvalues'].append(inputs['tranparams'][t])

            inputs['whitenoise'] = float(dictionary['whitenoise'])
            inputs['whitenoiselo'] = float(dictionary['whitenoiselo'])
            inputs['whitenoisehi'] = float(dictionary['whitenoisehi'])

        dtind = int(np.where(np.array(inputs['tranlabels']) == 'dt')[0])
        inputs['t0'] = inputs['toff'] + inputs['tranparams'][dtind]
        inputs['sigclip']     = float(dictionary['sigclip'])


        if self.n == 0:
            self.inputs['jointparams'] = dictionary['jointparams']
            if type(self.inputs['jointparams']) == str: self.inputs['jointparams'] = [dictionary['jointparams']]
            self.inputs['binlen']      = str_to_bool(dictionary['binlen'])
            self.inputs['timesTdur']   = float(dictionary['timesTdur'])

        #try: inputs['wave_gap'] = str_to_bool(dictionary['wave_gap'])
        #except(TypeError): inputs['wave_gap'] = [int(i) for i in dictionary['wave_gap']]

        try: inputs['midclip_inds'] = str_to_bool(dictionary['midclip_inds'])
        except(TypeError): inputs['midclip_inds'] = [int(i) for i in dictionary['midclip_inds']]

        if self.n == 0:
            self.inputs['samplecode'] = dictionary['samplecode']            
            if self.inputs['samplecode'] == 'dynesty': pass
                
            elif self.inputs['samplecode'] == 'emcee':
                self.inputs['nwalkers'] = int(dictionary['nwalkers'])
                self.inputs['nsteps']   = int(dictionary['nsteps'])
                self.inputs['burnin']   = int(dictionary['burnin'])

            self.inputs['optext']      = str_to_bool(dictionary['optext'])
            self.inputs['istarget']    = str_to_bool(dictionary['istarget'])
            self.inputs['isasymm']     = str_to_bool(dictionary['isasymm'])
            self.inputs['invvar']      = str_to_bool(dictionary['invvar'])
            self.inputs['dividewhite'] = str_to_bool(dictionary['dividewhite'])
            self.inputs['ldmodel']     = str_to_bool(dictionary['ldmodel'])
            self.inputs['fullsample']  = str_to_bool(dictionary['fullsample'])
            self.inputs['makeplots']   = str_to_bool(dictionary['makeplots'])
            if self.inputs['sysmodel'] == 'GP': self.inputs['dynestypool'] = str_to_bool(dictionary['dynestypool'])

        inputs['datacubepath']    = dictionary['datacubepath']
        inputs['specstretchpath'] = dictionary['specstretchpath']
        
        self.inputs[self.subdir] = inputs

    def equalizeArrays1D(self, unevenlist, padwith=0):
        # should work for nested arrays up to 2D

        maxarraycols = np.max([len(i) for i in unevenlist])
        self.numberofzeros = maxarraycols - [len(i) for i in unevenlist] 

        newarray = [np.append(array, np.zeros(self.numberofzeros[i])+padwith) for i, array in enumerate(unevenlist)] 
        numpyarray = np.hstack(newarray).reshape(len(unevenlist), maxarraycols) 

        return numpyarray

    def equalizeArrays2D(self, unevenlist, padwith=0):

        self.maxarraycols = np.max([len(i) for i in unevenlist])
        self.maxarrayrows = np.max(np.hstack([[len(j) for j in unevensublist] for unevensublist in unevenlist]))

        zeroarray = np.zeros((len(unevenlist), self.maxarraycols, self.maxarrayrows)) + padwith
        for i in range(len(unevenlist)):
            for j in range(len(unevenlist[i])):
                for k in range(len(unevenlist[i][j])):
                    zeroarray[i][j][k] = unevenlist[i][j][k]

        return zeroarray

