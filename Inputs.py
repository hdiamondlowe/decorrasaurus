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

        self.subdirectories = subdirectories

        for n, subdir in enumerate(self.subdirectories):
            self.n = n
            self.subdir = subdir
            if directoryname:
                self.speak('going into directory {0}'.format(directoryname[0]))
                self.directoryname = directoryname[0]
                self.readInputs()
            else: 
                self.createDirectory()
                self.readInputs()
        self.speak('successfully read in and joined input parameters')

    def createDirectory(self):
        ''' Create a new directory to put detrender stuff in'''

        self.speak('reading input file from {0}'.format(self.subdir))

        file = open(self.subdir+'/input.init')
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


        if self.n == 0:
            self.speak('creating new detrender directory')
            self.speak('where possible, input values from {0} will be used'.format(self.subdir))
            self.filename = dictionary['filename']
            self.nightname = [dictionary['nightname']]


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

        else: 
            self.nightname.append(dictionary['nightname'])
        
        self.speak('copying {0} to directory {1}'.format(self.subdir+'/input.init', self.directoryname))
        copyfile(self.subdir+'/input.init', self.directoryname+self.nightname[self.n]+'_input.init')
        #copyfile(self.subdir+'/'+self.starlist[self.n], self.directoryname+self.nightname[self.n]+'_'+self.starlist[self.n])
            

    def readInputs(self):

        inputfilenames = []

        for file in os.listdir(self.directoryname):
            if file.endswith('_input.init'):
                inputfilenames.append(file)

        # be careful here! if for some reason the sorted list of your directories does not match up with the sorted list of the observation nights, you will introduce an error
        inputfilenames = sorted(inputfilenames)#, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))

        self.speak('reading {0} file from {1}'.format(inputfilenames[self.n], self.directoryname))

        file = open(self.directoryname+inputfilenames[self.n])
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
            else:
                try: return float(s)
                except(ValueError): 
                    if s in dictionary.keys(): return float(dictionary[s])
                    else: return str(s)

        if self.n == 0:
            self.filename = dictionary['filename']
            self.saveas = self.directoryname
            self.nightname = [dictionary['nightname']]

        else:
            self.nightname.append(dictionary['nightname'])

        # target and comparison list now designated in mosasaurus cube

        if self.n == 0:
            self.against = dictionary['against']
            self.fitlabels = [dictionary['fitlabels']]
            if type(self.fitlabels[self.n]) == str: self.fitlabels[self.n] = [self.fitlabels[self.n]]
            self.polyfit = [int(dictionary['polyfit'])]
            self.polylabels = [[string.ascii_uppercase[x] for x in range(self.polyfit[self.n])]]

            self.Teff = float(dictionary['Teff'])
            self.Teff_unc = float(dictionary['Teff_unc'])
            self.logg = float(dictionary['logg'])
            self.logg_unc = float(dictionary['logg_unc'])
            self.z = float(dictionary['z'])
            self.z_unc = float(dictionary['z_unc'])
            self.ldlaw = str_to_bool(dictionary['ldlaw'])

            self.T0 = float(dictionary['T0'])
            self.P = float(dictionary['P'])
            self.Tdur = float(dictionary['Tdur'])
            self.inc = float(dictionary['inc'])
            self.a = float(dictionary['a'])
            self.ecc = float(dictionary['ecc'])
            self.omega = float(dictionary['omega'])
            self.epochnum = [int(dictionary['epochnum'])]
            self.toff = [self.T0 + self.P*self.epochnum[self.n]]

        else:
            #self.fitlabels.append(dictionary['fitlabels'])
            # quick hack to make testing go faster - all fit labels are the same for all nights so only set them for self.n = 0
            self.speak( 'using hack to set all fitlabels to be the same as those from dataset0')
            self.fitlabels.append(self.fitlabels[0])
            if type(self.fitlabels[self.n]) == str: self.fitlabels[self.n] = [self.fitlabels[self.n]]
            #self.polyfit.append(int(dictionary['polyfit']))
            # quick hack to make testing go faster - all fit labels are the same for all nights so only set them for self.n = 0
            self.polyfit.append(self.polyfit[0])
            self.polylabels.append([string.uppercase[x] for x in range(self.polyfit[self.n])])
            self.epochnum.append(int(dictionary['epochnum']))
            self.toff.append(self.T0 + self.P*self.epochnum[self.n])

        if self.n == 0:
            self.tranlabels = [dictionary['tranlabels']]
            self.tranparams = [[str_to_bool(i) for i in dictionary['tranparams']]]
            self.tranbounds = [[[str_to_bool(i) for i in dictionary['tranbounds_low']], [str_to_bool(i) for i in dictionary['tranbounds_high']]]]
            self.wavelength_lims = [float(i) for i in dictionary['wavelength_lims']]
            assert(len(self.tranbounds[self.n][0]) == len(self.tranlabels[self.n]) and len(self.tranbounds[self.n][1]) == len(self.tranlabels[self.n])), 'There is something wrong with the transit parameter bounds'

            self.fitparams = [[1 for f in self.fitlabels[self.n]]]
            self.polyparams = [[1 for p in self.polylabels[self.n]]]

            self.freeparambounds = [[], []]
            self.freeparamnames = []
            self.freeparamvalues = []
            for p, plabel in enumerate(self.polylabels[self.n]):
                self.freeparambounds[0].append(True)
                self.freeparambounds[1].append(True)
                self.freeparamnames.append(plabel+str(self.n))
                self.freeparamvalues.append(self.polyparams[self.n][p])
            for f, flabel in enumerate(self.fitlabels[self.n]):
                self.freeparambounds[0].append(True)
                self.freeparambounds[1].append(True)
                self.freeparamnames.append(flabel+str(self.n))
                self.freeparamvalues.append(self.fitparams[self.n][f])
            for t, tlabel in enumerate(self.tranlabels[self.n]):
                if type(self.tranbounds[self.n][0][t]) == bool and self.tranbounds[self.n][0][t] == False: continue
                if self.tranbounds[self.n][0][t] == 'Joint': continue
                self.freeparambounds[0].append(self.tranbounds[self.n][0][t])
                self.freeparambounds[1].append(self.tranbounds[self.n][1][t])
                self.freeparamnames.append(tlabel+str(self.n))
                self.freeparamvalues.append(self.tranparams[self.n][t])
            

            dtind = int(np.where(np.array(self.tranlabels[self.n]) == 'dt')[0])
            self.t0 = [self.toff[self.n] + self.tranparams[self.n][dtind]]

        else:
            self.tranlabels.append(dictionary['tranlabels'])
            self.tranparams.append([str_to_bool(i) for i in dictionary['tranparams']])
            self.tranbounds.append([[str_to_bool(i) for i in dictionary['tranbounds_low']], [str_to_bool(i) for i in dictionary['tranbounds_high']]])

            self.fitparams.append([1 for f in self.fitlabels[self.n]])
            self.polyparams.append([1 for p in self.polylabels[self.n]])

            for p, plabel in enumerate(self.polylabels[self.n]):
                self.freeparambounds[0].append(True)
                self.freeparambounds[1].append(True)
                self.freeparamnames.append(plabel+str(self.n))
                self.freeparamvalues.append(self.polyparams[self.n][p])
            for f, flabel in enumerate(self.fitlabels[self.n]):
                self.freeparambounds[0].append(True)
                self.freeparambounds[1].append(True)
                self.freeparamnames.append(flabel+str(self.n))
                self.freeparamvalues.append(self.fitparams[self.n][f])
            for t, tlabel in enumerate(self.tranlabels[self.n]):
                if type(self.tranbounds[self.n][0][t]) == bool and self.tranbounds[self.n][0][t] == False: continue
                if self.tranbounds[self.n][0][t] == 'Joint': continue
                self.freeparambounds[0].append(self.tranbounds[self.n][0][t])
                self.freeparambounds[1].append(self.tranbounds[self.n][1][t])
                self.freeparamnames.append(tlabel+str(self.n))
                self.freeparamvalues.append(self.tranparams[self.n][t])

            dtind = int(np.where(np.array(self.tranlabels[self.n]) == 'dt')[0])
            self.t0.append(self.toff[self.n] + self.tranparams[self.n][dtind])

        if self.n == 0:
            self.binlen = str_to_bool(dictionary['binlen'])
            self.sigclip = float(dictionary['sigclip'])
            self.timesTdur = float(dictionary['timesTdur'])
            try: self.midclip_inds = [str_to_bool(dictionary['midclip_inds'])]
            except(TypeError): self.midclip_inds = [[int(i) for i in dictionary['midclip_inds']]]

            self.samplecode = dictionary['samplecode']
            
            if self.samplecode == 'dynesty': pass
                
            elif self.samplecode == 'emcee':
                self.nwalkers = int(dictionary['nwalkers'])
                self.nsteps = int(dictionary['nsteps'])
                self.burnin = int(dictionary['burnin'])

            self.optext = str_to_bool(dictionary['optext'])
            self.istarget = str_to_bool(dictionary['istarget'])
            self.isasymm = str_to_bool(dictionary['isasymm'])
            self.invvar = str_to_bool(dictionary['invvar'])
            self.ldmodel = str_to_bool(dictionary['ldmodel'])
            self.domcmc = str_to_bool(dictionary['domcmc'])

            self.makeplots = str_to_bool(dictionary['makeplots'])

            self.datacubepath = [dictionary['datacubepath']]
        
        else:

            try: self.midclip_inds.append(str_to_bool(dictionary['midclip_inds']))
            except(TypeError): self.midclip_inds.append([int(i) for i in dictionary['midclip_inds']])

            self.datacubepath.append(dictionary['datacubepath'])


                    

