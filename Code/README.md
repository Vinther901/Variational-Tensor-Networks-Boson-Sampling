### [ script.py ](./script.py)
Contains the essential python class for performing gradient-based optimization of the variational MPS along with the parameterized local basis.

### [ DataGeneration.py ](./DataGeneration.py)
When run, it produces a folder 'GeneratedData' which is filled with data from simulations of the non Gaussian circuit with the parameterized local basis, similar to that found in the 'Data' folder of this repository. The settings for the experiments can be specified within the file.

### [ OverviewOfConsiderationsForParameterizedLBO.ipynb ](./OverviewOfConsiderationsForParameterizedLBO.ipynb)
Is a notebook that showcases the different considerations one can make when utilizing the parameterized LBO approach. It compares two different ways of optimizing the local basis to doing no basis optimization at all, and how this may be build on top of by standard LBO techniques. Then, two methods for reconstrucing the fock basis amplitudes from the optimized basis is discussed.

### [ VariationalSimulationOfGBS.ipynb ](./VariationalSimulationOfGBS.ipynb)
Is a notebook that walks through the basic steps for doing a variational simulation of a Gaussian boson sampling experiment. It also shows how to use the optimal local basis for GBS to achieve a better simulation at basically no computational overhead.