import json

from attrdict import AttrDict

from post_process.pp_plot import create_plot_bounds_training, create_plot_noise
from post_process.pp_compare_ELBO_Entropies import get_estimates
from post_process.pp_posterior_collapse import get_final_encoder_entropies, create_plot_posterior_collapse

# Path to the model with saved statistics
PATH_TO_FOLDER = "results/CelebA"

with open(f'{PATH_TO_FOLDER}/config.json') as json_file:
    config = json.load(json_file)
config = AttrDict(config)

# 1. Plot ELBO vs. SumOfEntropies throughout training, with close-up
print("1. Create plot 'fig_bounds_full'.", end=' ')
###
title = f'(a) {config.VAE_type} on {config.dataset_type}'
###
create_plot_bounds_training(PATH_TO_FOLDER, config, title)
print("Done.")

# # 2. Variance Reduction.
# Plot ELBO vs. SumOfEntropies for fully trained model (variance reduction)
# get and save the estimates, then plot them
print("2. Estimate lower bounds for variance comparisons.")
n_samples_list = [10, 100, 1000, 10000]
n_runs = 10
get_estimates(PATH_TO_FOLDER, config, n_samples_list, n_runs=n_runs)
###
title = f'(a) Variance Reduction ({config.VAE_type} on {config.dataset_type})'
###
create_plot_noise(PATH_TO_FOLDER, n_samples_list, n_runs=n_runs, title=title)

# 3. Posterior Collapse
# Plot the encoder entropies throughout training, then as a bar plot at the end of training.
print("3. Create plots for Posterior Collapse analysis.", end=' ')
###
title = f'(a) Posterior Collapse ({config.VAE_type} on {config.dataset_type})'
###
encoder_entropies = get_final_encoder_entropies(PATH_TO_FOLDER, config)
create_plot_posterior_collapse(PATH_TO_FOLDER, config, encoder_entropies, title)
print("Done.")
