Required Packages
-----------------

The code uses the following packages (with the tested version number in parentheses):
- python (3.7.9)
- numpy (1.19.1)
- scikit-learn (0.23.2)
- pytorch (1.6.0)
- torchvision (0.7)
- matplotlib (3.3.1)
- tqdm (4.50.0)


Repeating the Experiments
-------------------------

Figure 1:

To repeat the experiments (with new random seeds), run the following:
    VAE.py linear PCA
        to run the linear VAE on the randomly generated PCA data set
    VAE.py non-linear MNIST
        to run VAE-1 on the MNIST data set
    VAE.py sigma-z-full PCA-ring
        to run VAE-3 on the randomly generated PCA-ring data set

To gain means results over multiple runs, repeat the experiments with the linear VAE and VAE-1 10 times, and of VAE-3
100 times (or adjust the Visualize.ipynb accordingly).
The outputs will be stored in the outputs/ folder. To visualize the results, run the Visualize.ipynb notebook.

Figure 9:

To repeat the experiments (with new random seeds), run the following:
Fig2(a): run VAE-3 on the randomly generated PCA-ring data set 100 time :
VAE.py sigma-z-full PCA-ring
Fig2(b): Change the parameter in line 429 of the VAE.py to "batch_size" = 100, and again run 100 times:
VAE.py sigma-z-full PCA-ring
Fig2(c): Change the parameter in line 429 of the VAE.py back to "batch_size" = 2000 and the parameter in line 427 to "
lr" : 0.005, and again run 100 times:
VAE.py sigma-z-full PCA-ring
The outputs will be stored in the outputs/ folder. To visualize the results, run the Visualize.ipynb notebook.

Figure 10 & Animations:

To repeat these experiments and visualizations, run the z-space.py with the following arguments:
z-space.py linear PCA
to run and visualize the linear VAE on the randomly generated PCA data set
z-space.py non-linear MNIST
to run and visualize VAE-1 on the MNIST data set
z-space.py sigma-z-full PCA-ring
to run and visualize VAE-3 on the randomly generated PCA-ring data set
        
The outputs will be generated in the output/z-space/ folder.