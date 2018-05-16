

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

        self.fitmodel = []
        for n, night in enumerate(self.inputs.nightname):
            poly = []
            for plabel in self.inputs.polylabels[n]:
                paramind = int(np.where(np.array(self.inputs.freeparamnames) == plabel+str(n))[0])
                poly.append(self.params[paramind])
            N = len(poly)
            polymodel = 0
            while N > 0:
                polymodel = polymodel + poly[N-1]*(self.wavebin['compcube'][n]['bjd']-self.inputs.toff[n])**(N-1)
                N = N -1
            x = []
            for flabel in self.inputs.fitlabels[n]:
                paramind = int(np.where(np.array(self.inputs.freeparamnames) == flabel+str(n))[0])
                x.append(self.params[paramind]*self.wavebin['compcube'][n][flabel])
            parammodel = np.sum(x, 0)
            self.fitmodel.append(polymodel + parammodel + 1)


        tranvalues = []
        for n, night in enumerate(self.inputs.nightname):
            values = {}
            for t, tranlabel in enumerate(self.inputs.tranlabels[n]):
                if tranlabel+str(n) in self.inputs.freeparamnames:
                    paramind = int(np.where(np.array(self.inputs.freeparamnames) == tranlabel+str(n))[0])
                    # need to reparameterize t0 u0 and u1
                    if tranlabel == 'u0': 
                        if self.inputs.ldlaw == 'sq': values[tranlabel] = (75./34.)*self.params[paramind] + (45./34.)*self.params[paramind+1]
                        elif self.inputs.ldlaw == 'qd': values[tranlabel] = (2./5.)*self.params[paramind] + (1./5.)*self.params[paramind+1]
                    elif tranlabel == 'u1': 
                        if self.inputs.ldlaw == 'sq': values[tranlabel] = (45./34.)*self.params[paramind-1] - (75./34.)*self.params[paramind]
                        elif self.inputs.ldlaw == 'qd': values[tranlabel] = (1./5.)*self.params[paramind-1] - (2./5.)*self.params[paramind]
                    else: values[tranlabel] = self.params[paramind]
                elif self.inputs.tranbounds[n][0][t] == 'Joint':
                    jointset = int(self.inputs.tranbounds[n][1][t])
                    paramind = int(np.where(np.array(self.inputs.freeparamnames) == tranlabel+str(jointset))[0])
                    #values[tranlabel] = self.params[paramind]
                    values[tranlabel] = tranvalues[jointset][tranlabel]
                else: 
                    # need to reparameterize to u0 and u1 (these were set to v0 and v1 during the ldtkparams step of lmfitter
                    if tranlabel == 'u0': 
                        if self.inputs.ldlaw == 'sq': values[tranlabel] = (75./34.)*self.inputs.tranparams[n][t] + (45./34.)*self.inputs.tranparams[n][t+1]
                        elif self.inputs.ldlaw == 'qd': values[tranlabel] = (2./5.)*self.inputs.tranparams[n][t] + (1./5.)*self.inputs.tranparams[n][t+1]
                    elif tranlabel == 'u1': 
                        if self.inputs.ldlaw == 'sq': values[tranlabel] = (45./34.)*self.inputs.tranparams[n][t-1] - (75./34.)*self.inputs.tranparams[n][t]
                        elif self.inputs.ldlaw == 'qd': values[tranlabel] = (1./5.)*self.inputs.tranparams[n][t-1] - (2./5.)*self.inputs.tranparams[n][t]
                    else: values[tranlabel] = self.inputs.tranparams[n][t]   
                    #values[tranlabel] = self.inputs.tranparams[n][t]   
            tranvalues.append(values)

        #print [tranvalues[n]['dt'] for n in range(len(self.inputs.nightname))]
        #print self.params[0:11]

        if self.inputs.istarget == True and self.inputs.isasymm == False:
            self.batmanmodel = []
            for n, night in enumerate(self.inputs.nightname):
                batman = BatmanLC(times=self.wavebin['compcube'][n]['bjd'], t0=(self.inputs.toff[n]+tranvalues[n]['dt']), 
                                  rp=tranvalues[n]['rp'], per=tranvalues[n]['per'], inc=tranvalues[n]['inc'], a=tranvalues[n]['a'], ecc=tranvalues[n]['ecc'], omega=tranvalues[n]['omega'], 
                                  u0=tranvalues[n]['u0'], u1=tranvalues[n]['u1'], ldlaw=self.inputs.ldlaw)#, batmanfac=self.inputs.batmanfac)
                batmanmodel = batman.batman_model()
                if n == 0 and np.all(batmanmodel == 1.): self.speak('batman model returned all 1s')
                self.batmanmodel.append(batmanmodel)
        if self.inputs.istarget == True and self.inputs.isasymm == True:
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
        elif self.inputs.istarget == False: 
            self.batmanmodel = np.ones(len(self.wavebin['compcube']['bjd']))
        #if self.dividewhite == True: return self.fit_model*self.batman_model*self.Zwhite[self.binnedok]*self.Zlambdat[self.binnedok]
        return np.array(self.fitmodel)*np.array(self.batmanmodel)

