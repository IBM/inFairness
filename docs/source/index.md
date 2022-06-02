# inFairness

inFairness is a Python package that allows training individually fair PyTorch models


## Installation

<!--inFairness can be installed using `pip`:

```
pip install inFairness
```-->


If you wish to install the latest development version, you can install directly by cloning this repository:

```
git clone https://github.com/IBM/inFairness
cd inFairness
pip install -e .
```


## Features

inFairness supports training of individually fair models and includes the following components:

#### Algorithms

1. [Sensitive Set Invariance (SenSeI)](https://arxiv.org/abs/2006.14168)
2. [Sensitive Subspace Robustness (SenSR)](https://arxiv.org/abs/1907.00020)

#### Metrics

1. [Embedded Xenial Pair Logistic Regression Metric (EXPLORE)](https://proceedings.mlr.press/v119/mukherjee20a.html)
2. [SVD Sensitive Subspace Metric](https://arxiv.org/abs/1907.00020)

----------

## API Documentation

```{toctree}
:caption: Index

tutorial
examples
papers
```

```{toctree}
:caption: Package Reference

reference/index
development/index
changelog
GitHub Repository <https://github.com/IBM/inFairness>
```
