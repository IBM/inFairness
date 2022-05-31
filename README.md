[![Python3.8+](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)


# inFairness

inFairness is a Python package that allows training individually fair PyTorch models


## Installation

<!--inFairness can be installed using `pip`:

```
pip install inFairness
```-->


If you wish to install the latest development version, you can install directly by cloning this repository:

```
git clone <git repo url>
cd inFairness
pip install -e .
```



## Features

inFairness supports training of individually fair models and includes the following algorithms:

1. Sensitive Set Invariance (SenSeI): [Paper](https://arxiv.org/abs/2006.14168)


### Coming soon

We plan to extend the package by integrating the following features:
1. Implement [Sensitive Subspace Robustness (SenSR)](https://arxiv.org/abs/1907.00020) algorithm.
2. Implement additional individually fair distance metrics
3. Allowing auditing of models


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

Some of these examples include:
1. Individually fair sentiment classifier: [Notebook](./examples/sentiment-analysis/sentiment_analysis_demo_api.ipynb)

### Minimal example

First, you need to import the relevant packages

```
from inFairness import distances
from inFairness.fairalgo import SenSeI
```

The `inFairness.auditor.SenSeIAuditor` helps the fairness algorithm compute the adversarial example given the input, `inFairness.distances` implements various distance metrics on the input and the output spaces, and the `inFairness.fairalgo.SenSeI` is the individually fair algorithm that ties all the components together.

Thereafter, we specify the distances, and 


```[python]
distance_x = distances.SVDSensitiveSubspaceDistance(w=0.01, n_components=50, X_train=data)
distance_y = distances.EuclideanDistance()

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
