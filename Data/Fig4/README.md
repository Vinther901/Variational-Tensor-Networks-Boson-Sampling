## Loading the data:
```
GBS_N4_covs.json & GBS_N7_covs.json
```
contains a list of the pure state covariance matrices that define the Gaussian Boson Sampling problem for 4 and 7 modes respectively.\
They can be loaded with json:
```
import json
with open("GBS_N4_covs.json", "r") as file:
    N4_covs = json.load(file)
```
The indexing correspond to that found in the data files:
```
GBS_N4_data.csv & GBS_N7_data.csv
```
Which can be loaded with pandas:
```
import pandas as pd
N4_df = pd.read_csv("GBS_N4_data.csv")
```
The results are produced with a cutoff of $D=8$ for the Fock basis and a bond dimension of $\chi=16$ for the Matrix Product states.

## Glossary for the data columns:
The columns consists of 4 distinct groups related to the following:
 - Fidelity (F_...)
 - Variational energy (E_..., Evar_...)
 - Spectral Weight (SW_...)
 - Runtime (..._runtimes)

for the different states denoted by the following:
 - dmrg: refers to the MPS obtained from running DMRG on the variational Hamiltonian
 - GS: is the ground state of the variational Hamiltonian obtained with the usual linear algebra eigensolvers.
 - true: is the state obtained from the 'thewalrus' module (thewalrus.state_vector(...))
 - MPS: refers to the MPS obtained from the algorithm of [Oh et al.](https://zenodo.org/records/7992736) (<mark> as implemented in "../..")

 Lastly the file
 ```
 res_sums = np.load("res_sums.npy")
 ```
 contains the values corresponding to $\varepsilon_\chi \sim \sum_{n=1}^{N-1}\left(1 - \sum_{k\leq\chi_n}\left(\sigma^{(n)}_k\right)^2\right)$ as defined in the paper.