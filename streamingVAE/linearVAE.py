import math
import torch
from torch.nn import Linear, Module, Parameter
from torch.nn.functional import softplus
from torch.distributions import Normal, kl_divergence



pi = torch.Tensor([math.pi])


class LinearVAE(Module):
    def __init__(self, H, D, num_samples=100):
        super(LinearVAE, self).__init__()
        self.H = H
        self.D = D
        self.num_samples = num_samples  # number of samples to draw to estimate the gradient stochastically
        self.encoder = Linear(D, H)
        self.decoder = Linear(H, D)
        self.sigma_param = Parameter(torch.Tensor([0.]))
        self.tau_param = Parameter(torch.Tensor([0.0] * H))
        self.p_z = Normal(torch.Tensor([0.]), torch.Tensor([1.]))  # prior

    def forward(self, x):
        sigma = softplus(self.sigma_param)
        tau = softplus(self.tau_param)
        N, D = x.shape

        z_encoded = self.encoder(x)  # deterministic encoding
        q_z = Normal(z_encoded, tau)
        z_samples = q_z.rsample([self.num_samples])  # draw many reparametrized samples for stochastic gradient estimation
        p_x_z_mean = self.decoder(z_samples)
        p_x_z = Normal(p_x_z_mean, sigma)  # decoder distribution
        elbo = p_x_z.log_prob(x).mean(0).sum() \
                    - kl_divergence(q_z, self.p_z).sum()
        three_entropies_linear = self.three_entropies_linear(N, D, sigma, tau)
        return elbo, three_entropies_linear

    @staticmethod
    def three_entropies_linear(N, D, sigma, tau):
        """Three Entropies (accumulated) for linear VAE, Eq.(33).
        Args:
        N (int) : batch size
        D (int) : output dimensionality
        sigma (torch.tensor, size=(1,)) : decoder standard deviation
        tau (torch.tensor, size=(H,)
        : encoder standard deviation
        """
        return N*(-D/2*(torch.log(2*pi)+1)-D*torch.log(sigma)+torch.log(tau).sum())

    def BIC(self, loglik, N):
        num_parameters = self.H * self.D + 1
        return num_parameters * math.log(N) - 2 * loglik
