"""## Model

We implement this VAE with the classical choice of directly learning the log-variances denoted as
- `logsigma2`: $\log(\sigma^2)$ 
- `logalpha2`: $[\log(\alpha_h^2)]_{h \in \{1,\dots,H\}}$

With these choices the standard deviations are given as
- $\sigma$ = `exp(0.5 * logsigma2)`
- $\alpha_h$ = `exp(0.5 * logalpha2[h])`
"""

import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, config):
        super(VariationalAutoencoder, self).__init__()
        self.H = config.latent_dims
        self.D = config.data_dims
        self.dataset_type = config.dataset_type
        self.VAE_type = config.VAE_type
        self.n_hidden = config.n_hidden

        # define layers & activations
        self.encoder_hidden = nn.Linear(self.D, self.n_hidden)
        self.encoder_hidden2 = nn.Linear(self.n_hidden, 2 * self.H)
        self.encoder_mu = nn.Linear(2 * self.H, self.H)
        self.encoder_logvar = nn.Linear(2 * self.H, self.H)
        self.enc_nonlinearity = nn.LeakyReLU()

        self.encoder_logvar.bias.data.fill_(-3.0)

        self.decoder_hidden = nn.Linear(self.H, 2 * self.H)
        self.decoder_hidden2 = nn.Linear(2 * self.H, self.n_hidden)
        self.decoder_out = nn.Linear(self.n_hidden, self.D)
        self.dec_nonlinearity = nn.LeakyReLU()

        # decoder variances (log)
        if self.VAE_type == "VAE-0":  # log_var = 0 -> var = 1
            self.logsigma2 = nn.Parameter(torch.zeros(1), requires_grad=False)
        elif self.VAE_type == "VAE-1" or self.VAE_type == "VAE-2":
            # adapt initialization if applicable
            self.logsigma2 = nn.Parameter(- 9 * torch.ones(1), requires_grad=True)
        elif self.VAE_type == "VAE-3":
            self.logsigma2 = nn.Parameter(torch.zeros(self.D), requires_grad=True)
        else:
            raise AssertionError(f"VAE of type ''{self.VAE_type}'' not implemented.")

        if self.VAE_type == "VAE-2":
            # prior variance (log)
            self.logalpha2 = nn.Parameter(torch.ones(self.H), requires_grad=True)

    def encode(self, x):
        """
        Encode given data points x, predicting mean and (co-)variance in latent space.
        """
        x = self.enc_nonlinearity(self.encoder_hidden(x))
        x = self.enc_nonlinearity(self.encoder_hidden2(x))
        enc_mu = self.enc_nonlinearity(self.encoder_mu(x))
        enc_logvar = self.enc_nonlinearity(self.encoder_logvar(x))
        return enc_mu, enc_logvar

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Reparametrization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode given latent representations z.
        """
        state = self.dec_nonlinearity(self.decoder_hidden(z))
        x = self.dec_nonlinearity(self.decoder_hidden2(state))
        if self.dataset_type == "MNIST":
            probs = torch.sigmoid(self.decoder_out(x))
            return probs
        return self.dec_nonlinearity(self.decoder_out(x))

    def forward(self, x):
        """
        Forward pass of the VAE.
        """
        enc_mu, enc_logvar = self.encode(x)
        z = self.reparameterize(enc_mu, enc_logvar)
        dec_mu = self.decode(z)
        return dec_mu, enc_mu, enc_logvar


class VAE_MNIST(nn.Module):
    def __init__(self, config):
        super(VAE_MNIST, self).__init__()

        self.H = config.latent_dims
        self.D = config.data_dims
        self.dataset_type = config.dataset_type
        self.VAE_type = config.VAE_type
        # self.n_hidden = config.n_hidden

        n_hidden1 = 64
        n_hidden2 = 32

        # define layers & activations
        self.encoder_hidden = nn.Linear(self.D, n_hidden1)
        self.encoder_hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.encoder_mu = nn.Linear(n_hidden2, self.H)
        self.encoder_logvar = nn.Linear(n_hidden2, self.H)
        self.enc_nonlinearity = nn.LeakyReLU()

        self.decoder_hidden = nn.Linear(self.H, n_hidden2)
        self.decoder_hidden2 = nn.Linear(n_hidden2, n_hidden1)
        self.decoder_out = nn.Linear(n_hidden1, self.D)
        self.dec_nonlinearity = nn.LeakyReLU()

        # decoder variances (log)
        if self.VAE_type == "VAE-0":  # log_var = 0 -> var = 1
            self.logsigma2 = nn.Parameter(torch.zeros(1), requires_grad=False)
        elif self.VAE_type == "VAE-1" or self.VAE_type == "VAE-2":
            self.logsigma2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        elif self.VAE_type == "VAE-3":
            self.logsigma2 = nn.Parameter(torch.zeros(self.D), requires_grad=True)
        else:
            raise AssertionError(f"VAE of type ''{self.VAE_type}'' not implemented.")

        if self.VAE_type == "VAE-2":
            # prior variance (log)
            self.logalpha2 = nn.Parameter(torch.ones(self.H), requires_grad=True)

    def encode(self, x):
        x = self.enc_nonlinearity(self.encoder_hidden(x))
        x = self.enc_nonlinearity(self.encoder_hidden2(x))
        enc_mu = self.enc_nonlinearity(self.encoder_mu(x))
        enc_logvar = self.enc_nonlinearity(self.encoder_logvar(x))
        return enc_mu, enc_logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        state = self.dec_nonlinearity(self.decoder_hidden(z))
        x = self.dec_nonlinearity(self.decoder_hidden2(state))
        if self.dataset_type == "MNIST":
            probs = torch.sigmoid(self.decoder_out(x))
            return probs
        return self.dec_nonlinearity(self.decoder_out(x))

    def forward(self, x):
        enc_mu, enc_logvar = self.encode(x)
        z = self.reparameterize(enc_mu, enc_logvar)
        dec_mu = self.decode(z)
        return dec_mu, enc_mu, enc_logvar
