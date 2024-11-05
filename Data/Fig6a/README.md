## Loading the data:
The '.csv' files can be loaded with pandas whereas the '.json' can be loaded with json (more details can be found in the [Fig4 folder](../Fig4))

## Glossary for the data:
Regarding the naming convention of the data files:
- 'N' refer to the number of modes
- 'D' is cutoff of the Fock basis
- 'chi' is the maximum bond dimension for the Matrix Product States
- 'k' is the value of the parameter $\kappa$ for the fully connected CZ-gate

**'basisparams_...'** contains the learned basis parameters from the parameterized Local Basis Optimization algorithm (pLBO) in the following format:
```
basisparams_...["basis parameter"][# sweep][# site]
```
i.e. basis parameter is one of [r, phi, theta, s, gamma, kappa, alpha_real, alpha_imag]
and the rows index the learning step
whereas the columns index the site

**data_...** contains the results in the form of the mean 'E_...' and variance 'Evar_...' of the variational Hamiltonian for the different states, including the norm ('SW_...') of the state. It also includes the runtimes of the algorithms ('..._runtimes').

The states are:
- GS: The ground state of the variational Hamiltonian in the Fock basis obtained with the usual linear algebra eigensolvers.
- MPS: The Matrix Product State obtained with our pLBO algorithm.