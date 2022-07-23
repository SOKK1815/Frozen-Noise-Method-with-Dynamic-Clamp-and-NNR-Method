''' input.py
    
    This file contains the input class that generates the hidden state and the input theory.

    The method is described in the following paper:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). 
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. 
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.
'''
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

class Input():
    ''' Class that generates the input to the ANN (hidden state) and to the model neuron (input theory).

    NOTE time (dt,T) in ms, freq in Hz, but qon en qoff in MHz
    '''
    def __init__(self): 
        # For all
        self.dt = None
        self.T = None          
        self.fHandle = [None, None]
        self.seed = None
        self.input = None
        self.p0 = None
        
        #for explicit E/I ratio distributions
        self.frac_inh = None
        self.frac_exc = None
        self.f_i = None
        self.f_e = None
        self.b = None
        self.dist = None #takes either 'normal' or 'lognorm'
        


        # For Markov models
        self.ron = None
        self.roff = None
        self.qon = []
        self.qoff = []
        self.kernel = None
        self.kerneltau = None
        self.xseed = None
        self.x = None
        self.xfix = None
        self.weight = []
        self.theta = None

    # Get dependend variables
    def get_tvec(self):
        '''Generate tvec and save the length
        '''
        self.tvec = np.arange(self.dt, self.T+self.dt, self.dt)
        self.length = len(self.tvec)

    def generate(self):
        '''Generate input and x from fHandle.
        '''
        if not self.fHandle:
            print('fHandle isn''t provided object')
        else:
            [self.input, self.x] = self.fHandle     

    def get_tau(self):
        '''Generates tau based on the hidden state switch rate
           i.e. ron/roff
        '''
        if self.ron == None or self.roff == None:
            print('Tau not defined, missing ron/roff')
        else:
            self.tau = 1/(self.ron+self.roff)
        
    def get_p0(self):
        '''Generates the probability of finding the hidden state
           in the 'ON' state.
        '''
        if self.ron == None or self.roff == None:
            print('P0 not defined, missing ron/roff')
        else:
            self.p0 = self.ron/(self.ron+self.roff)   

    def get_theta(self):
        '''Generates the firing rate differences.
        '''
        if self.qon == [] or self.qoff == []:
            print('Theta not defined, missing qon/qoff')
        else:
            self.theta = sum(self.qon-self.qoff)
    
    def get_w(self):
        '''Generates the weight matrix based on qon/qoff.
        '''
        if self.qon == [] or self.qoff == []:
            print('Weight not defined, missing qon/qoff')
        else:
            self.w = np.log(self.qon/self.qoff)
    def get_b(self):
        if self.frac_inh == None or self.frac_exc == None:
            if self.dist != None:
                
                print('b not defined, missing frac_inh/frac_exc')
        else:
            self.b = self.frac_inh * self.f_i/ (self.frac_exc * self.f_e)

    def get_all(self):
        '''Runs all the functions to create dependent variables.
        '''
        self.get_tvec()
        self.generate()
        self.get_tau()
        self.get_p0()
        self.get_theta()
        self.get_w()
        self.get_b()


    @staticmethod
    def create_qonqoff(mutheta, N, alphan, regime, qseed=None):
        ''' Generates [qon, qoff] with qon and qoff being a matrix filled with the 
            firing rate of each neuron based on the hidden state.

            INPUT
            mutheta (float): the summed difference between qon and qoff
            N (int): number of neurons in the ANN
            alphan (?): ? 
            regime (int): coincedence of push-pull regime
            qseed (int): seed to set the random number generator (rng)

            OUTPUT
            [qon, qoff]: array containing the firing rates of the neurons during both states
        '''
        # Sample qon and qoff from a rng.
        np.random.seed(qseed)
        qoff = np.random.randn(N, 1) 
        qon = np.random.randn(N, 1)

        if N > 1:
            # Creates a q distribution with a standard deviation of 1 
            qoff = qoff/np.std(qoff)
            qon = qon/np.std(qon)
        qoff = qoff - np.mean(qoff)
        qon = qon - np.mean(qon)

        if regime == 1:   
            # Coincedence regime !! No E/I balance, little negative weights
            qoff = (alphan*qoff+1)*mutheta/N
            qon = (alphan*qon+2)*mutheta/N
        else:
            # Push-pull regime !! E/I balance, negative weights
            qoff = (alphan*qoff+1)*mutheta/np.sqrt(N)
            qon = (alphan*qon+1+1/np.sqrt(N))*mutheta/np.sqrt(N)
        
        # Set all negative firing rates to absolute value
        qoff[qoff<0] = abs(qoff[qoff<0])
        qon[qon<0] = abs(qon[qon<0])

        return [qon, qoff]
    

    @staticmethod
    def create_qonqoff_balanced(N,  meanq, stdq, qseed=None):
        ''' Generates normally distributed [qon, qoff] with qon and qoff 
            being a matrix filled with the firing rate of each neuron based 
            on the hidden state.

            INPUT
            N (int): number of neurons in the ANN
            meanq (float): mean of the normal distribution from which q is sampled
            stdq (float): standard deviation of the normal distribution
            qseed (int): seed to set the random number generator (rng)

            OUTPUT
            [qon, qoff]: array containing the firing rates of the neurons during both states
        '''
        # Sample qon and qoff from a rng.
        np.random.seed(qseed)
        qoff = np.random.randn(N, 1)
        qon = np.random.randn(N, 1)

        # Consider the normal distribution
        if N > 1: 
            qoff = qoff/np.std(qoff)
            qon = qon/np.std(qon)
        qoff = stdq*(qoff-np.mean(qoff))+meanq
        qon = stdq*(qon-np.mean(qon))+meanq

        # Set all negative firing rates to absolute value
        qoff[qoff<0] = abs(qoff[qoff<0])
        qon[qon<0] = abs(qon[qon<0])
        
        return [qon, qoff]


    @staticmethod
    def create_qonqoff_balanced_uniform(N, minq, maxq, qseed=None):
        '''Generates uniformly distributed [qon, qoff] with qon and qoff 
           being a matrix filled with the firing rate of each neuron based 
           on the hidden state.

            INPUT
            N (int): number of neurons in the ANN
            minq (float): minimal firing rate
            maxq (float): maximal firing rate
            qseed (int): seed to set the random number generator (rng)

            OUTPUT
            [qon, qoff]: array containing the firing rates of the neurons during both states
        '''
        # Sample qon and qoff from a rng
        np.random.seed(qseed)
        qoff = np.random.rand(N, 1)
        qon = np.random.rand(N, 1)

        # Consider the uniform distribution
        qoff = minq + np.multiply((maxq-minq), qoff)
        qon = minq + np.multiply((maxq-minq), qon)

        return [qon, qoff]
    

    def generate_mu_std(self, mu_inh_on, alphas):
        '''
        Generates the sets of mean values and standard deviations of the
        distributions for excitatory and inhibitory neurons during the on and off state
        when the Neuron Number Ratio is made explicit.
        
        INPUT
        
        mu_inh_on (float): chosen mean firing rate of inhibitory neurons during on state, must be less than f_inh
        alphas (array): the coefficients of variation of the four distribution

        OUTPUT
        mu_set (array): array of the means of the four distributions of firing rates q_on and q_off for excitatory and inhibitory neurons
        std_set (array): array of the standard deviations of the four distributions of firing rates q_on and q_off for excitatory and inhibitory neurons

        '''
        self.get_p0()
        self.get_b()
        mu_inh_off = (self.f_i - mu_inh_on * self.p0)/(1 - self.p0)
        alpha_inh_on, alpha_inh_off, alpha_exc_on, alpha_exc_off = alphas
        mu_i_ratfunc = mu_inh_off/mu_inh_on
        
        
        if self.dist == 'lognorm':
            
            l_rand = ((1 + alpha_inh_on**2)/(1 + alpha_inh_off**2))**(self.b/2) * ((1 + alpha_exc_on**2)/(1 + alpha_exc_off**2))**(1/2)
            mu_e_ratfunc = mu_i_ratfunc**self.b * l_rand
        
        elif self.dist == 'normal':
            d_alpha_i = alpha_inh_on**2 - alpha_inh_off**2
            d_alpha_e = alpha_exc_on**2 - alpha_inh_off**2
            
            mu_e_ratfunc = mu_i_ratfunc **self.b * np.exp(self.b * d_alpha_i + d_alpha_e)
            
        else:
            raise SyntaxError('No distribution type specified')
        
        mu_exc_on = mu_e_ratfunc * self.f_e/(1 + (mu_e_ratfunc - 1) * self.p0)
        mu_exc_off = mu_exc_on/mu_e_ratfunc
        
        std_i_on = alphas[0] * mu_inh_on
        std_i_off = alphas[1] * mu_inh_off
        std_e_on = alphas[2] * mu_exc_on
        std_e_off = alphas[3] * mu_exc_off
        
        mu_set = [mu_inh_on, mu_inh_off, mu_exc_on, mu_exc_off]
        std_set = [std_i_on, std_i_off, std_e_on, std_e_off]
        
        return mu_set, std_set
    
    
    def create_qonqoff_ratioed(self, N, mu_set, std_set, qseed=None):
        ''' Generates either normally or lognormally distributed [qon, qoff] for a given NNR (E/I) in the ANN.
        
        INPUT
        N (int): number of neurons in the ANN
        mu_set (array): array of four mean firing rates during the on and off 
        states of the hidden state for both inhibitory and excitatory neurons.
        std_set (array): array of four standard deviations of the four firing rate distributions
        qseed (int): seed to set the random number generator (rng)

        OUTPUT
        [qon, qoff]: array containing the firing rates of the neurons during both states


        '''
        
        mu_inh_on, mu_inh_off, mu_exc_on, mu_exc_off = mu_set
        std_inh_on, std_inh_off, std_exc_on, std_exc_off = std_set

        mu_w = np.empty(len(mu_set))
        std_w = np.empty(len(mu_set))
        
        
        #generate the mean weights for the lognormal distribution
        for dis in range(len(mu_set)):
            mu_w[dis]= np.log(mu_set[dis]**2/np.sqrt(mu_set[dis]**2 + std_set[dis]**2))
            std_w[dis] = np.log(1 + std_set[dis]**2/mu_set[dis]**2)

        rng = default_rng(qseed)
        
        #generate firing rates
        if self.dist == 'lognorm':
            q_inh_on = rng.lognormal(mu_w[0], std_w[0], int(self.frac_inh * N))
            q_inh_off = rng.lognormal(mu_w[1], std_w[1], int(self.frac_inh * N))
        
            q_exc_on = rng.lognormal(mu_w[2], std_w[2], int(self.frac_exc * N))
            q_exc_off = rng.lognormal(mu_w[3], std_w[3], int(self.frac_exc * N))
            
        elif self.dist == 'normal':
            
            q_inh_on = rng.normal(mu_inh_on, std_inh_on, int(self.frac_inh * N))
            q_inh_off = rng.normal(mu_inh_off, std_inh_off, int(self.frac_inh * N))
            
            q_exc_on = rng.normal(mu_exc_on, std_exc_on, int(self.frac_exc * N))
            q_exc_off = rng.normal(mu_exc_off, std_exc_off, int(self.frac_exc * N))
            
        else:
            raise SyntaxError('No distribution type specified')
            
            
        qon = np.concatenate([q_inh_on, q_exc_on])
        qoff = np.concatenate([q_inh_off, q_exc_off])
        
        
        qoff[qoff<0] = abs(qoff[qoff<0])#irrelevant for lognormal dist
        qon[qon<0] = abs(qon[qon<0])
        
        
        
        return [qon, qoff]
    
    
    



    def markov_hiddenstate(self): 
        ''' Takes ron and roff from class object and generates
            the hiddenstate if xfix is empty.
        '''
        np.random.seed(self.xseed)
        
        # Generate x
        if self.xfix == None:
            self.get_p0()
            xs = np.zeros(np.shape(self.tvec)) 

            #Initial value 
            i = np.random.rand()
            if i < self.p0:
                xs[0] = 1
            else:
                xs[0] = 0

            # Make x
            for n in np.arange(1, self.length): 
                i = np.random.rand()
                if xs[n-1] == 1: 
                    if i < self.roff*self.dt:
                        xs[n] = 0
                    else:
                        xs[n] = 1
                else: 
                    if i < self.ron*self.dt:
                        xs[n] = 1
                    else:
                        xs[n] = 0
        else:
            xs = self.xfix

        return xs


    def markov_input(self, dynamic=False, raster=False):
        ''' Takes qon, qoff and hiddenstate and generates input.
            Optionally when dynamic is a dictinary of g0_values it
            generates a conductance over time based on the hidden state. 
            It can also return input to create raster plot.
        '''
        xs = self.x
        nt = self.length 
        w = np.log(self.qon/self.qoff) 

        if dynamic:
            ni = dynamic.keys()
        else:
            ni = range(len(self.qon))

        # Make spike trains (implicit)
        stsum = np.zeros((nt, 1))
        if self.kernel != None:
            if self.kernel == 'exponential':
                tfilt = np.arange(0, 5*self.kerneltau+self.dt, self.dt)
                kernelf = np.exp(-tfilt/self.kerneltau)
                kernelf = kernelf/(self.dt*sum(kernelf)) 
            elif self.kernel == 'delta':
                kernelf = 1./self.dt
        
        xon = np.where(xs==1)
        xoff = np.where(xs==0)
        np.random.seed(self.seed)
        spike_trains = []
        
        # Create the input generated by the artificial neural network
        for k in ni:
            randon = np.random.rand(np.shape(xon)[0],np.shape(xon)[1])
            randoff = np.random.rand(np.shape(xoff)[0], np.shape(xoff)[1])
            sttemp = np.zeros((nt, 1))
            sttempon = np.zeros(np.shape(xon))
            sttempoff = np.zeros(np.shape(xoff))

            sttempon[randon < self.qon[k]*self.dt] = 1.
            sttempoff[randoff < self.qoff[k]*self.dt] = 1.
            
            sttemp[xon] = np.transpose(sttempon)
            sttemp[xoff] = np.transpose(sttempoff)

            if dynamic:
                stsum = stsum + dynamic[k]*sttemp
            else:
                stsum = stsum + w[k]*sttemp 
            
            spike_trains.append(sttemp)
            # #SanityCheck for individual spikes
            #plt.plot(sttemp)
            #plt.show()

        if self.kernel != None:
            stsum = np.convolve(stsum.flatten(), kernelf, mode='full')

        stsum = stsum[0:nt]
        ip = stsum 
        
        #return the spike trains of all neurons in the ANN to make raster plot
        if raster==True:
            return spike_trains
        
        #return regular frozen noise input
        else:
            return ip
        