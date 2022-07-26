''' make_dynamic_experiments.py

    This file contains the function that generates a hidden state and the corresponding theoretical
    input (current or conductance).

    The method is described in the following paper:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). 
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. 
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.

    Dynamic clamp adaptation is described in:
    Schutte, M. and Zeldenrust, F. (2021) Increased neural information transfer for a conductance input:
    a dynamic clamp approach to study information flow. Msc. University of Amsterdam. Available at: https://scripties.uba.uva.nl
    
    NOTE Make sure that you save the hidden state & input theory with the experiments, it is
    essential for the information calculation!
'''
from foundations.input import Input
from foundations.dynamic_clamp import get_g0
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def make_dynamic_experiments(qon_qoff_type, baseline, tau, factor_ron_roff,
                             mean_firing_rate, sampling_rate, duration,
                             alpha_set=None, frac_inh=None, frac_exc=None,
                             mu_inh_on=None, distribution=None, mutheta=None, return_raster=False, seed=None):
    ''' Make hidden state and let an ANN generate a theoretical input corresponding to that hidden state.

    INPUT
    qon_qoff_type (str): The method of qon/qoff generation normal, balanced or balanced_uniform
    baseline (int): Baseline for scaling the input current in picoampere
    tau (ms): Switching speed of the hidden state in milliseconds
    factor_on_off (float): ratio determining the occurance of the ON and OFF state 
    mean_firing_rate (int): Mean firing rate of the artificial neurons in kilohertz
    sampling rate (int): Sampling rate of the experimental setup (injected current) in kilohertz
    duration (float): Length of the duration in milliseconds
    alpha_set (array, optional): Only needed if qon_qoff_type == 'balanced_ratioed' set of four floats defining the the narrowness of the qon and qoff distributions for a defined ratio of inhibitory and excitatory neurons in the ANN
    frac_inh (float, optional): Fraction of inhibitory neurons in ANN, only needed if qon_qoff_type == 'balanced_ratioed'
    frac_exc (float, optional): Fraction of excitatory neurons in ANN, only needed if qon_qoff_type == 'balanced_ratioed'
    mu_inh_on (float, optional): The mean firing rate of the inhibitory neurons during the on-state in kHz. Only relevant if qon_qoff_type == 'balanced_ratioed'
    distribution (str): Type of distribution from which the qons and qoffs are taken
    mutheta (float): The summed difference between qon and qoff. Generally chosen to be 1. Otherwise it needs to be computed. It is returned as value.
    return_raster (boolean): Decides wether the function returns the spiking patterns of all neurons in the ANN.
    seed (optional): seed used in the random number generator

    OUTPUT
    [input_theory, dynamic_theory, hidden_state, weights, theta, spikes] (array): array containing theoretical input and hidden state
    input_theory (array): the theoretical current input
    dynamic_theory (array): the theoretical conductance input
    hidden_state: 1xN array with hidden state values 0=OFF 1=ON
    weights (array): weights of neurons in ANN
    theta (float): squared sum of all q_on/q_off values, independent of mutheta
    spikes (array): Nx(duration/dt) array of spike instances of all neurons N 0=no spike, 1=spike
    '''
    # Set RNG seed, if no seed is provided
    if seed == None:
        np.random.seed()
        seed = np.random.randint(1000000000)

    # Fixed parameters
    N = 1000
    dt = 1./sampling_rate
    tau_exponential_kernel = 5
    alpha = np.sqrt(1/8)            # SEM * N
    stdq = alpha*mean_firing_rate
    ron = 1./(tau*(1+factor_ron_roff))
    roff = factor_ron_roff*ron
    v_rest = -65
    Er_exc, Er_inh = (0, -75)

    # Create input from artifical network
    input_bayes = Input()
    input_bayes.dt = dt
    input_bayes.T = duration
    input_bayes.kernel = 'exponential'
    input_bayes.kerneltau = tau_exponential_kernel
    input_bayes.ron = ron
    input_bayes.roff = roff
    input_bayes.seed = seed
    input_bayes.xseed = seed

    # Create qon/qoff
    if qon_qoff_type == 'normal':
        # mutheta = mutheta             #The summed difference between qon and qoff
        alphan = alpha
        regime = 1
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff(
            mutheta, N, alphan, regime, seed)
    elif qon_qoff_type == 'balanced':
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_balanced(
            N, mean_firing_rate, stdq, seed)
    elif qon_qoff_type == 'balanced_uniform':
        minq = 10
        maxq = 100
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_balanced_uniform(
            N, minq, maxq, seed)
    elif qon_qoff_type == 'balanced_ratioed':

        input_bayes.f_e = 6.7/1000
        input_bayes.f_i = 24.7/1000
        input_bayes.frac_inh = frac_inh
        input_bayes.frac_exc = frac_exc
        input_bayes.dist = distribution
        
        mu_set, std_set = input_bayes.generate_mu_std(mu_inh_on, alpha_set)
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_ratioed(N, mu_set, std_set, seed)
    else:
        raise SyntaxError('No qon/qoff creation type specified')

    # Generate weights and hiddenstate
    input_bayes.get_all()
    input_bayes.x = input_bayes.markov_hiddenstate()
    input_bayes.weight = input_bayes.w
    

    # Generate exc and inh
    g0_exc, g0_inh = get_g0(v_rest, input_bayes.w, Er_exc, Er_inh)
    g_exc = input_bayes.markov_input(g0_exc)
    g_inh = input_bayes.markov_input(g0_inh)
    dynamic_theory = (g_exc, g_inh)
    
    

    # Generate input_current for comparison
    input_theory = input_bayes.markov_input()
    
    theta = input_bayes.theta
       

    # #SanityCheck for input (Vm=-40) and hiddenstate
    # fig, axs = plt.subplots(2, figsize=(12,12))
    # fig.suptitle('Dynamic Clamp conductances')

    # for idx, val in enumerate(input_bayes.x):
    #     if val == 1:
    #         axs[0].axvline(idx, c='lightgray')
    #         axs[1].axvline(idx, c='lightgray')

    # axs[0].plot(g_exc, c='red')
    # axs[0].set(ylabel='Exc. conductance [mS]')

    # axs[1].plot(g_inh, c='blue')
    # axs[1].set(ylabel='Inh. conductance [mS]')

    # plt.show()
    
    if return_raster==True:
        
        input_spiketrains = input_bayes.markov_input(raster=True)
        
        return [input_theory, dynamic_theory, input_bayes.x, input_bayes.weight, theta, input_spiketrains]
    
    else:
        return [input_theory, dynamic_theory, input_bayes.x, input_bayes.weight, theta]

    

    