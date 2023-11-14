import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

from getdist import loadMCSamples, plots, mcsamples, MCSamples
from tensiometer_dev.tensiometer import *

def plot_frac_fisher(prior_chain=MCSamples(), posterior_chain=MCSamples(), params=[], labels=[], 
                     log=False, print_improvement=False, norm=True):
    # Non-logarithmic effective number of parameters
    print(f'Non-log N_eff \t= {gaussian_tension.get_Neff(posterior_chain, prior_chain=prior_chain, param_names=params):.5}\n')
    KL_param_names = params
    # Logarithmic transformation
    if log:
        # Take the logarithm of the parameters: KL decomposition -> power-law decomposition
        log_params = []
        log_labels = []
        for name, label in zip(params, labels):
            for chain in [prior_chain, posterior_chain]:
                p = chain.getParams()
                method = getattr(p, name)
                chain.addDerived(np.log(method), name='log_'+name, label='\\log '+label)
            log_params += ['log_'+name]
            log_labels += ['\\log '+label]
        # Logarithmic effective number of parameters
        print(f'Non-log N_eff \t= {gaussian_tension.get_Neff(posterior_chain, prior_chain, param_names=log_params):.5}\n')
        KL_param_names = log_params
    
    # Compute the KL modes
    KL_eig, KL_eigv, KL_param_names = gaussian_tension.Q_UDM_KL_components(prior_chain, posterior_chain, 
                                                                        param_names=KL_param_names)
    
    if print_improvement:
        with np.printoptions(precision=2, suppress=True):
            if any(eig < 1. for eig in KL_eig):
                print('Improvement factor over the prior:\n', KL_eig)
                print('Discarding error units due to negative values.')
            else:
                print('Improvement factor over the prior in E.U.:\n', np.sqrt(KL_eig-1))

    # Compute the fractional Fisher matrix
    KL_param_names, KL_eig, fractional_fisher, _ = gaussian_tension.Q_UDM_fisher_components(prior_chain, posterior_chain, 
                                                                                            KL_param_names, 
                                                                                            which='1')
    # Use alternative version (normalized column sum)
    if norm:
        dict = gaussian_tension.linear_CPCA_chains(prior_chain, posterior_chain, param_names=params)
        KL_eig, KL_eigv, fractional_fisher = dict['CPCA_eig'], dict['CPCA_eigv'], dict['CPCA_var_contributions']

    # Plot (showing values and names)
    figsize = int(len(params)/1.6)
    plt.figure(figsize=(figsize, figsize))
    im1 = plt.imshow(fractional_fisher, cmap='viridis')
    num_params = len(fractional_fisher)
    # The following loop is used to display the fractional fisher values inside the cells
    for i in range(num_params):
        for j in range(num_params):
            if fractional_fisher[j,i]>0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(fractional_fisher[j,i],2), va='center', ha='center', color=col)
    plt.xlabel('KL mode (error improvement)');
    plt.ylabel('Parameters');
    ticks  = np.arange(num_params)
    if any(eig < 1. for eig in KL_eig):
        labels = [str(t+1)+'\n'+str(l) for t,l in zip(ticks, np.round(KL_eig, 2))]
    else:
        labels = [str(t+1)+'\n'+str(l) for t,l in zip(ticks, np.round(np.sqrt(KL_eig-1), 2))]
    plt.xticks(ticks, labels, horizontalalignment='center');
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in KL_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right');

    return KL_eig, KL_eigv, KL_param_names