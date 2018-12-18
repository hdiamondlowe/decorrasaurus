

from imports import *
from BatmanLC import BatmanLC

class ModelMaker(Talker):

    def __init__(self, inputs, wavebin, params):
        ''' initialize the model maker'''
        Talker.__init__(self)

        self.inputs = inputs
        self.wavebin = wavebin
        self.params = params
    
    def makemodel(self):

        self.fitmodel = {}
        for subdir in self.wavebin['subdirectories']:
            n = self.inputs[subdir]['n']
            poly = []
            for plabel in self.inputs[subdir]['polylabels']:
                paramind = np.argwhere(self.wavebin['freeparamnames'] == plabel+str(n))[0][0]
                poly.append(self.params[paramind])
            N = len(poly)
            polymodel = 0
            while N > 0:
                polymodel += poly[N-1]*(self.wavebin[subdir]['compcube']['bjd']-self.inputs[subdir]['toff'])**(N-1)
                N -= 1
            x = []
            for flabel in self.inputs[subdir]['fitlabels']:
                paramind = np.argwhere(np.array(self.wavebin['freeparamnames']) == flabel+str(n))[0][0]
                x.append(self.params[paramind]*self.wavebin[subdir]['compcube'][flabel])
            parammodel = np.sum(x, 0)
            self.fitmodel[subdir] = (polymodel + parammodel + 1)


        tranvalues = {}
        for subdir in self.wavebin['subdirectories']:
            tranvalues[subdir] = {}
            n = self.inputs[subdir]['n']
            for t, tranlabel in enumerate(self.inputs[subdir]['tranlabels']):
                if tranlabel+str(n) in self.wavebin['freeparamnames']:
                    paramind = np.argwhere(self.wavebin['freeparamnames'] == tranlabel+str(n))[0][0]
                    # need to reparameterize t0 u0 and u1
                    if tranlabel == 'u0': 
                        if self.inputs['ldlaw'] == 'sq': tranvalues[subdir][tranlabel] = (75./34.)*self.params[paramind] + (45./34.)*self.params[paramind+1]
                        elif self.inputs['ldlaw'] == 'qd': tranvalues[subdir][tranlabel] = (2./5.)*self.params[paramind] + (1./5.)*self.params[paramind+1]
                    elif tranlabel == 'u1': 
                        if self.inputs['ldlaw'] == 'sq': tranvalues[subdir][tranlabel] = (45./34.)*self.params[paramind-1] - (75./34.)*self.params[paramind]
                        elif self.inputs['ldlaw'] == 'qd': tranvalues[subdir][tranlabel] = (1./5.)*self.params[paramind-1] - (2./5.)*self.params[paramind]
                    else: tranvalues[subdir][tranlabel] = self.params[paramind]
                elif tranlabel in self.inputs['jointparams']:
                    jointset = self.wavebin['subdirectories'][0]
                    jointind = np.argwhere(np.array(self.inputs['subdirectories']) == jointset)[0][0]
                    paramind =  np.argwhere(np.array(self.wavebin['freeparamnames']) == tranlabel+str(jointind))[0][0]
                    tranvalues[subdir][tranlabel] = self.params[paramind]
                    #tranvalues[subdir][tranlabel] = tranvalues[subdir][tranlabel]
                else: 
                    # need to reparameterize to u0 and u1 (these were set to v0 and v1 during the ldtkparams step of lmfitter
                    if tranlabel == 'u0': 
                        if self.inputs['ldlaw'] == 'sq': tranvalues[subdir][tranlabel] = (75./34.)*self.inputs[subdir]['tranparams'][t] + (45./34.)*self.inputs[subdir]['tranparams'][t+1]
                        elif self.inputs['ldlaw'] == 'qd': tranvalues[subdir][tranlabel] = (2./5.)*self.inputs[subdir]['tranparams'][t] + (1./5.)*self.inputs[subdir]['tranparams'][t+1]
                    elif tranlabel == 'u1': 
                        if self.inputs['ldlaw'] == 'sq': tranvalues[subdir][tranlabel] = (45./34.)*self.inputs[subdir]['tranparams'][t-1] - (75./34.)*self.inputs[subdir]['tranparams'][t]
                        elif self.inputs['ldlaw'] == 'qd': tranvalues[subdir][tranlabel] = (1./5.)*self.inputs[subdir]['tranparams'][t-1] - (2./5.)*self.inputs[subdir]['tranparams'][t]
                    else: tranvalues[subdir][tranlabel] = self.inputs[subdir]['tranparams'][t]   
                    #values[tranlabel] = self.inputs.tranparams[n][t]   

        #print [tranvalues[n]['dt'] for n in range(len(self.inputs.nightname))]

        # make the transit model with batman; or some alternative
        if self.inputs['istarget'] and not self.inputs['isasymm']:
            self.batmanmodel = {}
            for n, subdir in enumerate(self.wavebin['subdirectories']):
                batman = BatmanLC(times=self.wavebin[subdir]['compcube']['bjd'], t0=(self.inputs[subdir]['toff']+tranvalues[subdir]['dt']), 
                                  rp=tranvalues[subdir]['rp'], per=tranvalues[subdir]['per'], inc=tranvalues[subdir]['inc'], a=tranvalues[subdir]['a'], ecc=tranvalues[subdir]['ecc'], omega=tranvalues[subdir]['omega'], 
                                  u0=tranvalues[subdir]['u0'], u1=tranvalues[subdir]['u1'], ldlaw=self.inputs['ldlaw'])#, batmanfac=self.inputs.batmanfac)
                batmanmodel = batman.batman_model()
                if n == 0 and np.all(batmanmodel == 1.): self.speak('batman model returned all 1s')
                self.batmanmodel[subdir] = batmanmodel
        elif self.inputs['istarget'] and self.inputs['isasymm']:
            rp, tau0, tau1, tau2 = [], [], [], []
            numtau = 0
            for k in tranvalues.keys():
                if 'tau' in k: numtau += 1
            numdips = numtau/3
            alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
            for i in range(numdips):
                rp.append(tranvalues['rp'+alphabet[i]])
                tau0.append(tranvalues['tau0'+alphabet[i]])
                tau1.append(tranvalues['tau1'+alphabet[i]])
                tau2.append(tranvalues['tau2'+alphabet[i]])
            t, F = self.wavebin['compcube']['bjd']-self.toff-tranvalues['dt'], tranvalues['F']
            for i in range(len(tau0)):
                F -= 2.*rp[i] * (np.exp((t-tau0[i])/tau2[i]) + np.exp(-(t-tau0[i])/tau1[i]))**(-1)
            self.batmanmodel = F
        elif not self.inputs['istarget']:
            self.batmanmodel = []
            for n, subdir in enumerate(self.wavebin['subdirectories']):
                self.batmanmodel.append(np.ones_like(self.wavebin['compcube'][n]['bjd']))

        # models to return
        if self.inputs['dividewhite'] and self.inputs['binlen']!='all': 
            # dont' think this really works anymore!
            return np.array(self.fitmodel)*np.array(self.batmanmodel)*np.array(self.wavebin['Zwhite'])*np.array(self.wavebin['Zcomp'])
        else: 
            fullmodel = {}
            for subdir in self.wavebin['subdirectories']:
                fullmodel[subdir] = self.fitmodel[subdir] * self.batmanmodel[subdir]
            return fullmodel

