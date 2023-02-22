import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamingdata
from linearVAE import LinearVAE

parser = argparse.ArgumentParser(description='Streaming Linear VAE')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
batch_size = args.batch_size
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.manual_seed(args.seed)


def train(model, dataloader, optimizer):
    model.train()
    elbos = []
    entropies = []
    gradmagnitude = []
    for batch_idx, (_, data,) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        elbo, three_entropies_linear = model(data)
        elbos.append(elbo.item())
        entropies.append(three_entropies_linear.item())
        loss = -elbo
        loss.backward()
        optimizer.step()
        gradmagnitude.append(torch.linalg.norm(torch.cat(list(torch.flatten(param.grad.clone()) for param in model.parameters()))))
        if batch_idx % args.log_interval == 0:
            print('Batch {} \t ELBO: {:.6f} \t Entropies: {:.6f}'.format(
                batch_idx,
                elbo.item() / len(data),
                three_entropies_linear.item() / len(data)))
    return elbos, entropies, gradmagnitude


def experiment_online_model_selection():
    outdir = "./output/modelselection"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    D = 10
    N = 100000
    a = np.random.normal(size=(D, D))
    u, s, _ = np.linalg.svd(a)
    noise_scale = 0.1
   
    datasets = [] 
    for H in (6, 2, 4):
        scale = np.linspace(1, 5, num=10)
        scale[H:] = 0
        datasets.append(streamingdata.StationaryMultivarGaussianDataset(N=N, D=D, scale=scale, transform=u, noise_scale=noise_scale))
    
    ds = torch.utils.data.ChainDataset(datasets)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    model_H = (2, 4, 6)
    model_elbo = []
    model_elbo_BIC = []
    model_entropies = []
    model_entropies_BIC = []
    model_gradmagnitude = []
    for H in model_H:
        model = LinearVAE(H=H, D=D).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        
        elbo, entropies, gradmagnitude = train(model, dataloader, optimizer)
        elbo = np.array(elbo)
        entropies = np.array(entropies)
        model_elbo.append(elbo)
        model_elbo_BIC.append(model.BIC(loglik=elbo, N=batch_size))
        model_entropies.append(entropies)
        model_entropies_BIC.append(model.BIC(loglik=entropies, N=batch_size))
        model_gradmagnitude.append(gradmagnitude)

    
    for H, elbo, elbo_BIC, entropies, entropies_BIC, gradient in zip(model_H, model_elbo, model_elbo_BIC, model_entropies, model_entropies_BIC, model_gradmagnitude):
        np.savetxt(os.path.join(outdir, "elbo-{}.txt".format(H)), elbo)
        np.savetxt(os.path.join(outdir, "elbo-BIC-{}.txt".format(H)), elbo_BIC)
        np.savetxt(os.path.join(outdir, "entropies-{}.txt".format(H)), entropies)
        np.savetxt(os.path.join(outdir, "entropies-BIC-{}.txt".format(H)), entropies_BIC)
        np.savetxt(os.path.join(outdir, "gradient-{}.txt".format(H)), gradient)



if __name__ == "__main__":
    experiment_online_model_selection()
    from figures import simple_plots
    simple_plots()