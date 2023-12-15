import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

from getdist import loadMCSamples, plots, mcsamples, MCSamples
from tensiometer_dev.tensiometer import *

# Plot the fractional Fisher Matrix, along with the effective number of parameters
# Arguments:
#   - prior_chain = MCSamples() object containing the prior distribution chain
#   - posterior_chain = MCSamples() object containing the posterior distribution chain
#   - params, labels = string arrays containing parameters' names and labels
#   - log = whether to apply a logarithmic transformation to the input chains
#   - print_improvement = whether to print the improvement factor of the posterior over the prior
#   - norm = whether to use the CPCA decomposition for the Fisher Matrix, in order to have rows and columns 
#     with normalized sum
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
        print(f'Log N_eff \t= {gaussian_tension.get_Neff(posterior_chain, prior_chain, param_names=log_params):.5}\n')
        KL_param_names = log_params
    
    # Compute the KL modes
    KL_eig, KL_eigv, KL_param_names = gaussian_tension.Q_UDM_KL_components(prior_chain, posterior_chain, 
                                                                        param_names=KL_param_names)
    
    if print_improvement:
        with np.printoptions(precision=2, suppress=True):
            if any(eig < 1. for eig in KL_eig):
                print('Improvement factor over the prior:', KL_eig)
                print('Discarding error units due to negative values.')
            else:
                print('Improvement factor over the prior in E.U.:\n', np.sqrt(KL_eig-1))

    # Compute the fractional Fisher matrix
    KL_param_names, KL_eig, fractional_fisher, _ = gaussian_tension.Q_UDM_fisher_components(prior_chain, posterior_chain, 
                                                                                            KL_param_names, 
                                                                                            which='1')
    # Use alternative version (normalized column sum)
    # Eigenvalues and eigenvectors of the KL-decomposed Fisher matrix are not changed
    if norm:
        dict = gaussian_tension.linear_CPCA_chains(prior_chain, posterior_chain, param_names=params)
        fractional_fisher = dict['CPCA_var_contributions']

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
    plt.xticks(ticks, labels, horizontalalignment='center', rotation=30);
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in KL_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right');

    return KL_eig, KL_eigv, KL_param_names

from python_scripts.prior import *
# Build the full MCSamples() prior chain
# Arguments:
#   - n = size of the chains,
#   - root_dir = full path to the directory containing the log.param file and the chains
#   - chain_name = prefix used to name the chain files
#   - ignore_rows = burn-in fraction 
#   - name_tag = name of the prior chain to be displayed on the plots
def full_prior(n=10000, root_dir='', chain_name='', config_name='', ignore_rows=0.3, name_tag='', include_class=True):
    prior = priorChain(n=n, root_dir=root_dir, chain_name=chain_name)
    # If the cosmo_prior chain has already been computed and saved, there's no need to do it again
    # Remember that, if you change the parameter or the configuration file, the prior distribution
    # needs to change as well, therefore you can remove these files. 
    # However, if you vary any of the parameter's prior, you should also repeat the posterior sampling
    if os.path.exists(root_dir+'cosmo_prior_.txt'):
        pass
    else:
        cosmo_prior = prior.get_dv_prior(include_class=include_class)
        cosmo_prior.saveAsText(root=root_dir+'/cosmo_prior_')

    try:
        cosmo_prior = loadMCSamples(file_root=root_dir+'/cosmo_prior_', no_cache=True)
    except FileNotFoundError:
        print(f'The chain files named {root_dir}/cosmo_prior_[...] can not be found!')
        return 0, 0

    # Nuisance Parameters (varying): repeat the same size and burn-in fraction used for the cosmo_prior object
    try:
        nuisance_prior = prior.get_nuisance_prior(config_name=config_name, ignore_rows=0.3)
    except FileNotFoundError:
        print(f'The config file named {root_dir+config_name}.yaml can not be found!')
        return 0, 0
    # Check completeness
    print(f'List of \"cosmological\" parameters: {cosmo_prior.getParamNames().getRunningNames()}')
    print(f'List of \"derived\"      parameters: {cosmo_prior.getParamNames().getDerivedNames()}')
    print(f'List of \"nuisance\"     parameters: {nuisance_prior.getParamNames().list()}')

    # Loading both cosmological and EFT parameters
    prior_chain = cosmo_prior.copy()
    p = nuisance_prior.getParams()
    for name, label in zip(prior.nuisance_names, prior.nuisance_labels):
        method = getattr(p, name)
        prior_chain.addDerived(method, name=name, label=label)
    prior_chain.name_tag = name_tag
    # Return param_limits and prior chain
    param_limits = prior.get_param_limits()
    params = list(param_limits)
    limits = list(param_limits.values())
    print('\nFixed Parameter Limits:')
    print(create_table(params, limits))
    return param_limits, prior_chain

# Create a nice-looking table to store multiple values from arrays with the same length.
# Values on the same row have the same index in the lists. 
def create_table(*lists):
    # Ensure all lists have the same length
    list_lengths = [len(lst) for lst in lists]
    if len(set(list_lengths)) != 1:
        return "Error: lists are not of the same length"

    # Combine all lists into one list of tuples
    combined_lists = list(zip(*lists))

    # Find the maximum length for each column (that is, each individual list)
    max_lengths = [max(len(str(item)) for item in list) for list in lists]

    # Create the table
    table = ''
    for row in combined_lists:
        # Add elements for each row, adjusting the width for each column
        table += ' | '.join(f'{str(item):<{max_len}}' for item, max_len in zip(row, max_lengths)) + '\n'

    return table

# Alternative to numpy.where(), meant to be used with non-numpy arrays and string arrays.
def find_index(list, condition):
    # List comprehension is the faster procedure
    return [i for i, elem in enumerate(list) if condition(elem)]