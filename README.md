[![Build Status](https://app.travis-ci.com/IBM/inFairness.svg?branch=main)](https://app.travis-ci.com/IBM/inFairness)
[![Python3.8+](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache-yellow?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)


# inFairness

inFairness is a PyTorch package that allows training individually fair PyTorch models


## Installation

inFairness can be installed using `pip`:

```
pip install inFairness
```


Alternatively, if you wish to install the latest development version, you can install directly by cloning this repository:

```
git clone <git repo url>
cd inFairness
pip install -e .
```



## Features

inFairness currently supports:

1. Training of individually fair models : [[Docs]](https://ibm.github.io/inFairness/reference/algorithms.html)
2. Auditing pre-trained ML models for individual fairness : [[Docs]](https://ibm.github.io/inFairness/reference/auditors.html)


### Coming soon

We plan to extend the package by integrating the following features:
1. Post-processing for Individual Fairness : [[Paper]](https://arxiv.org/abs/2110.13796)
2. Support Individually fair boosting and ranking
3. Additional individually fair metrics


## Contributing

We welcome contributions from the community in any form - whether it is through the contribution of a new fair algorithm, fair metric, a new use-case, or simply reporting an issue or enhancement in the package. To contribute code to the package, please follow the following steps:

1. Clone this git repository to your local system
2. Setup your system by installing dependencies as: `pip3 install -r requirements.txt` and `pip3 install -r  build_requirements.txt`
3. Add your code contribution to the package. Please refer to the [`inFairness`](./inFairness) folder for an overview of the directory structure
4. Add appropriate unit tests in the [`tests`](./tests) folder
5. Once you are ready to commit code, check for the following:
   1. Coding style compliance using: `flake8 inFairness/`. This command will list all stylistic violations found in the code. Please try to fix as much as you can
   2. Ensure all the test cases pass using: `coverage run --source inFairness -m pytest tests/`. All unit tests need to pass to be able merge code in the package.
6. Finally, commit your code and raise a Pull Request.


## Tutorials

The [`examples`](./examples) folder contains tutorials from different fields illustrating how to use the package.

### Minimal example

First, you need to import the relevant packages

```
from inFairness import distances
from inFairness.fairalgo import SenSeI
```

The `inFairness.distances` module implements various distance metrics on the input and the output spaces, and the `inFairness.fairalgo` implements various individually fair learning algorithms with `SenSeI` being one particular algorithm.

Thereafter, we instantiate and fit the distance metrics on the training data, and 


```[python]
distance_x = distances.SVDSensitiveSubspaceDistance()
distance_y = distances.EuclideanDistance()

distance_x.fit(X_train=data, n_components=50)

# Finally instantiate the fair algorithm
fairalgo = SenSeI(network, distance_x, distance_y, lossfn, rho=1.0, eps=1e-3, lr=0.01, auditor_nsteps=100, auditor_lr=0.1)
```

Finally, you can train the `fairalgo` as you would train your standard PyTorch deep neural network:

```
fairalgo.train()

for epoch in range(EPOCHS):
    for x, y in train_dl:
        optimizer.zero_grad()
        result = fairalgo(x, y)
        result.loss.backward()
        optimizer.step()
```


##  Authors

- Mikhail Yurochkin
- Mayank Agarwal
- Aldo Pareja
- Onkar Bhardwaj
