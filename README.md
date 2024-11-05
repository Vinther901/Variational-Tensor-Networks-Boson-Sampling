# <mark> The repository is under construction.</mark>
<mark>(In the meantime, access to my personal repository can be granted upon request.)</mark>

This repository accompanies the paper [arXiv:2410.18740](https://arxiv.org/abs/2410.18740) - Jonas Vinther & Michael J. Kastoryano:
# "Variational Tensor Network Simulation of Gaussian Boson Sampling and Beyond"

## Repository Structure:
1. [ Data ](#data)
    - Fig2
    - Fig4
    - Fig5
    - Fig6a
    - Fig6bc
    - Fig7
2. [ Code ](#code)
    - ...
    
[[ Installation ](#installation) | [ Citation ](#citation) | [ License ](#license) | [ Acknowledgments ](#acknowledgments) ]

<a name="data"></a>
## 1. Data
Contains the data related to the figures of the paper and the necessary steps for reconstructing the results.

<a name="code"></a>
## 2. Code
Contains an overview of the scripts developed for the project and how to use them.

<a name="installation"></a>
## <mark> Installation
**Environment Requirements**:
Install environment:
```
conda env create -f environment.yml
```
This will create a conda environment with all the necessary packages. Make sure to activate the environment:
```
conda activate VarTN
```

<a name="citation"></a>
## Citation
If you find our code or paper helpful, please consider citing:
```
@misc{vinther2024variationaltensornetworksimulation,
      title={Variational Tensor Network Simulation of Gaussian Boson Sampling and Beyond}, 
      author={Jonas Vinther and Michael James Kastoryano},
      year={2024},
      eprint={2410.18740},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2410.18740}, 
}
```

<a name="license"></a>
## License

This software is released under the MIT License. You can view a license summary [here](LICENSE).

Portions of source code are taken from external sources under different licenses, including the following:
- [BosonSupremacy](https://github.com/sss441803/BosonSupremacy) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7992736.svg)](https://doi.org/10.5281/zenodo.7992736)


- [The Walrus](https://github.com/XanaduAI/thewalrus) (Apache License, Version 2.0)
- [Strawberry Fields](https://github.com/XanaduAI/strawberryfields) (Apache License, Version 2.0)

<a name="acknowledgments"></a>
## Acknowledgements
We acknowledge support from the AWS Center for
Quantum Computing, the Carlsberg foundation and
from the Novo Nordisk Foundation, Grant number
NNF22SA0081175, Quantum Computing Programme