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
    - script.py
    - DataGeneration.py
    - OverviewOfConsiderationsForParameterizedLBO.ipynb
    - VariationalSimulationOfGBS.ipynb
3. [ Extension to Boson sampling ](#nonCVextension)
    
[[ Installation ](#installation) | [ Citation ](#citation) | [ License ](#license) | [ Acknowledgments ](#acknowledgments) ]

<a name="data"></a>
## 1. [ Data ](./Data)
Contains the data related to the figures of the paper and the necessary steps for reconstructing the results.

<a name="code"></a>
## 2. [ Code ](./Code)
Contains an overview of the scripts developed for the project and how to use them.

<a name="nonCVextension"></a>
## 3. [ Extension to Boson sampling ](./Extension_to_Boson_Sampling.ipynb)
Is a notebook with a simple example of how to use our tool for Boson sampling. Evidently it is commonly referred to as [ Scattershot Boson sampling ](https://strawberryfields.ai/photonics/demos/run_scattershot_bs.html).

<a name="installation"></a>
## Installation
**Environment Requirements**:
An overview of the environment used for development is found in 'requirements.txt' and to install this environment run:
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