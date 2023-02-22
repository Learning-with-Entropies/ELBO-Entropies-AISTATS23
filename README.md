# ELBO & 3 Entropies

This repository complements the theoretical investigations in the AISTATS2023 paper "The ELBO of Variational Autoencoders Converges to a Sum of Entropies".

---

When training standard Variational Autoencoders (VAEs) the central learning objective, the Evidence Lower BOund (ELBO),  converges to the sum of three Entropies, or simply

<img src="https://latex.codecogs.com/gif.latex?\text{ELBO}(\Theta,\Phi)=-\mathcal{H}_\text{dec}(\Theta,\Phi)%2B\frac{1}{N}\sum_n\mathcal{H}_\text{enc}^{(n)}(\Phi)-\mathcal{H}_\text{prior}"/> 

Note that this requires Gaussian VAEs with learnable Decoder
variance <img src="https://latex.codecogs.com/gif.latex?\sigma^2"/>.
For the theoretical statement, the required assumption and generalizations see the attached paper (Theorems 1 to 3).

With this repository you can reproduce the experiments in the paper.
Moreover, you can train Gaussian VAEs and investigate the proposed entropy-based perspective on VAE training which, e.g., allows for investigating posterior collapse. The following features are currently implemented:

- `VAE_type`, all VAEs are defined on data domain <img src="https://latex.codecogs.com/gif.latex?\mathcal{X}=\mathbb{R}^D"/> using a latent
  space <img src="https://latex.codecogs.com/gif.latex?\mathcal{Z}=\mathbb{R}^H"/>.
  - `VAE-0`: Gaussian VAE with standard normal prior and **fixed** Decoder variance <img src="https://latex.codecogs.com/gif.latex?\sigma^2=1"/>
  - `VAE-1`: Gaussian VAE with standard normal prior and **learnable** Decoder variance <img src="https://latex.codecogs.com/gif.latex?\sigma^2"/>
  - `VAE-2`: Gaussian VAE with **learnable** normal prior and **learnable** Decoder variance <img src="https://latex.codecogs.com/gif.latex?\sigma^2"/> (see `weight_constraint`)
  - `VAE-3`: Gaussian VAE with standard normal prior and **diagonal, latent-depending covariance
    matrix** <img src="https://latex.codecogs.com/gif.latex?\Sigma(z)"/>  for the Decoder
- `dataset_type`
  - image data sets (please provide the directory to config)
    - `CELEBA` (download CelebA data set [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
    - `MNIST`
  - high-energy physics data set `SUSY`, download [here](https://archive.ics.uci.edu/ml/datasets/SUSY).
  - synthetic data sets from low(er) dimensional manifolds, perturbed observations with Gaussian noise
    - Manifolds in 2D: `cirlce`, `8gauss`, `cosine`, `eight` (lemniscate of Gerono)
    - Manifolds in 3D: `swiss_roll`, `spiral_3D`
    - Manifold in arbitrary dimension: `manifold` with hyperparameters `k` = dimensionality of manifold, `D` =
      dimensionality of data space
- `weight_constraint` (required for `VAE-2`) from {`rescale`,`regularize`}: Enforce the weight constraint on first linear layer of decoder net with columns <img src="https://latex.codecogs.com/gif.latex?W_h"/> via
  - `rescale`: the norm of each column <img src="https://latex.codecogs.com/gif.latex?W_h"/> will be set to one in each iteration
  - `regularize`: the training objective will be extended to also
    minimize <img src="https://latex.codecogs.com/gif.latex?\gamma\sum\nolimits_{h}(W_h^TW_h-1)^2">
- `KL_annealing` (Boolean): If `True` simple linear annealing is applied to the weight of the KL term in the ELBO.

---

## Pre-requisites.

This repo requires Python>=3.8.0 along with the packages listed in requirements.txt,
which you can install directly after cloning this directory using

```
pip install -r requirements.txt
```

---
Note that when using Python 3.10 the package `attrdict` requires a small fix as some packages have been moved.

```
from collections.abc import Mapping
```

We currently work on substituting the package.

---

## How to use this repository.

For Streaming Applications with Linear VAEs see the code in [streamingVAE](streamingVAE).
For the pure verification experiments see also the code and instructions in [verification](verification).

To train a VAE model and analyze it using the derived entropy-based perspective proceed as follows:

- Prepare the config dictionary (in `main.py`), setting `VAE_type`, data set, learning rates etc.
- Do not forget to provide the correct `config.data_dir` to the data directory. No data directory is required for the
  artificial
  manifold data.
- All relevant figures are tracked using `tensorboard` throughout training (call `tensorboard --logdir=LOGDIR`).
- **After training** run the script `post_process.py` with the path to your trained model to create the plots.

---