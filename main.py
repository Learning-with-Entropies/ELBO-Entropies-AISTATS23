"""
This is the main file to run the experiments.
Modify the config file as desired (see README.md for available options).
"""

# handy dictionary access & saving
# attrdict is deprecated and not working under Python 3.10
# Fix: from collections.abc import Mapping
from attrdict import AttrDict

import json

# imports from this repo
from prepare_data import get_dataloader
from model import VariationalAutoencoder, VAE_MNIST
from train import train
from visualize import *

from model_CelebA import *
# tensorboard for logging
from torch.utils.tensorboard import SummaryWriter

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Preliminaries & Config

config = {
    "dataset_type": "SUSY",
    "data_dir": "/path/to/SUSY/SUSY.csv",
    "logdir": "results/SUSY",
    "VAE_type": "VAE-1",
    "batch_size": 1024,
    "img_dim": 64,  # MNIST: 28, CELEBA: 64
    "latent_dims": 12,  # latent dimension H
    "n_hidden": 128,  # not relevant for MNIST & CelebA (they have separate models)
    "epochs": 150,  # number of epochs
    "n_epochs_log": 10,  # evaluation & logging
    "n_epochs_vis": 10,  # visualizations
    "lr_regular": 0.001,
    "lr_fast": 0.001,  # higher lr for variance parameters
    "KL_annealing": False,
    "beta": 1.0,
    "optimal_vars": False,
    "device": device}

config = AttrDict(config)

# Extend config, if applicable
if config.VAE_type == "VAE-2":
    config.weight_constraint = "rescale"

if config.dataset_type == "manifold":
    config.manifold_dims = 10  # k = dimensional manifold
    config.data_dims = 100  # D = dimension of ambient data space ( dim(X)=D )

# Integration of Tensorboard
writer = SummaryWriter(config.logdir)

# Get the data
print("Load & prepare the data...")
train_dataloader, test_dataloader = get_dataloader(config)

# Create model & start training
print("Create the model...")
if config.dataset_type == "MNIST":
    vae = VAE_MNIST(config=config).to(device)
elif config.dataset_type == "CELEBA":
    vae = VanillaVAE(config=config).to(device)
else:
    vae = VariationalAutoencoder(config=config).to(device)

ELBO, ELBO_held_out, three_entropies, H_enc_h_hist = train(vae, train_dataloader, test_dataloader, config=config,
                                                           writer=writer)

# save results
with open(f'{config.logdir}/config.json', 'w') as f:
    json.dump(config, f)
torch.save(vae.state_dict(), config.logdir + '/vae')
np.savetxt(config.logdir + '/bounds.txt', ELBO)
np.savetxt(config.logdir + '/bounds_held_out.txt', ELBO_held_out)
np.savetxt(config.logdir + '/three_entropies.txt', three_entropies)
np.save(config.logdir + '/H_enc_h_hist.npy', H_enc_h_hist)

# Plot & Evaluation of the latent space
create_training_plot(ELBO, ELBO_held_out, three_entropies, config)
plot_latent_violin(vae, train_dataloader, config)

writer.close()
