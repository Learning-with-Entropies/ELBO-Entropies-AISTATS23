import json

import numpy as np
# handy dictionary access & saving
import pandas as pd
from attrdict import AttrDict
from tqdm import tqdm

from elbo import lower_bound
from model import VariationalAutoencoder, VAE_MNIST
from model_CelebA import *
from prepare_data import get_dataloader


def compare_ELBO_3Entropies(config, vae, n_batches, runs, data_loader):
    """
    Simple function to calculate mean of ELBO & 3 Entropies for n_batches batches for multiple runs.
    :param config: config file
    :param vae: vae loaded from state_dict
    :param n_batches: number of batches
    :param runs: number of estimations
    :param data_loader: data_loader instance
    """
    # lists to store averaged values over multiple runs
    ELBO_runs = []
    H_sum_runs = []
    log_prob_runs = []
    kl_div_runs = []
    H_prior_runs = []
    H_enc_runs = []
    H_dec_runs = []

    for run in tqdm(range(runs)):
        # lists to accumulate values
        ELBO_acc = 0
        H_sum_acc = 0
        log_prob_acc = 0
        kl_div_acc = 0
        H_prior_acc = 0
        H_enc_acc = 0
        H_dec_acc = 0

        for n_iter, data_batch in enumerate(data_loader):
            # Preprocess batch (if necessary) & send to device
            if config.dataset_type == "MNIST":
                data_batch = data_batch[0]  # discard labels for MNIST
                data_batch = data_batch.view(-1, config.data_dims)  # flatten images
            data_batch = data_batch.to(config.device)

            # calculate ELBO & Sum of entropies
            recon_batch, enc_mu, enc_logvar = vae(data_batch)
            ELBO, log_prob, kl_div, H_prior, H_enc, H_dec = lower_bound(vae, data_batch, recon_batch, enc_mu,
                                                                        enc_logvar, config, beta=1.0)
            # Calculate the Sum of Entropies
            H_sum = - H_prior - H_dec + H_enc

            # divide values for this batch (all are means over the current batch, cf. lower_bound)
            ELBO_acc += ELBO.cpu().detach().numpy()
            H_sum_acc += H_sum.cpu().detach().numpy()
            log_prob_acc += log_prob.cpu().detach().numpy()
            kl_div_acc += kl_div.cpu().detach().numpy()
            H_prior_acc += H_prior.cpu().detach().numpy()
            H_enc_acc += H_enc.cpu().detach().numpy()
            H_dec_acc += H_dec.cpu().detach().numpy()

            if n_iter + 1 == n_batches:
                ELBO_runs.append(ELBO_acc / (n_iter + 1))
                H_sum_runs.append(H_sum_acc / (n_iter + 1))
                log_prob_runs.append(log_prob_acc / (n_iter + 1))
                kl_div_runs.append(kl_div_acc / (n_iter + 1))
                H_prior_runs.append(H_prior_acc / (n_iter + 1))
                H_enc_runs.append(H_enc_acc / (n_iter + 1))
                H_dec_runs.append(H_dec_acc / (n_iter + 1))
                break
    return ELBO_runs, H_sum_runs, log_prob_runs, kl_div_runs, H_prior_runs, H_enc_runs, H_dec_runs


# Run the post-convergence analysis.
# Save all essential values, estimated on the given number of batches, over 10 runs.

def get_estimates(PATH_TO_FOLDER, config, n_samples_list, n_runs):
    # # load config file
    # with open(f'{PATH_TO_FOLDER}/config.json') as json_file:
    #     config = json.load(json_file)
    # config = AttrDict(config)
    # config.device = 'cpu'  # switch to cpu calculations here

    # set up model class and load state dict
    if config.dataset_type == "MNIST":
        vae = VAE_MNIST(config=config).to(config.device)
    elif config.dataset_type == "CELEBA":
        vae = VanillaVAE(config=config).to(config.device)
    else:
        vae = VariationalAutoencoder(config=config).to(config.device)

    # load state dict and set to eval-mode
    vae.load_state_dict(torch.load(f"{PATH_TO_FOLDER}/vae"))
    vae.eval()

    # Select the number of samples & number of runs
    for n_samples in n_samples_list:
        print(f"Estimating {n_runs} times with {n_samples} samples...")
        # simple trick: call data_loader with changed batch_size
        batch_size = 10  # works for every data set
        n_batches = n_samples // batch_size
        assert n_samples % batch_size == 0
        config.batch_size = batch_size

        train_dataloader, test_dataloader = get_dataloader(config, test_shuffle=True)

        ELBO_runs, H_sum_runs, log_prob_runs, kl_div_runs, H_prior_runs, H_enc_runs, H_dec_runs = \
            compare_ELBO_3Entropies(config, vae, n_batches, n_runs, test_dataloader)

        # save dataframe for Table 1
        df = pd.DataFrame({'H_sum': H_sum_runs,
                           'ELBO': ELBO_runs,
                           'reg_score_H': [H_enc_runs[i] - H_prior_runs[i] for i in range(n_runs)],
                           'reg_score_conv': [- kl_div_runs[i] for i in range(n_runs)],
                           'recon_score_H': [- H_dec_runs[i] for i in range(n_runs)],
                           'recon_score_conv': log_prob_runs,
                           'kl_div': kl_div_runs,
                           'H_prior': H_prior_runs,
                           'H_enc': H_enc_runs,
                           'H_dec': H_dec_runs
                           })

        df.to_csv(PATH_TO_FOLDER + f"/estimates_{n_samples}.csv", index=False)
    return
