# inFairness


inFairness is a Python package that allows training individually fair PyTorch models

[![Source Code](https://img.shields.io/badge/Code-GitHub-g?color=blueviolet)](https://github.com/IBM/inFairness)
[![Build Status](https://app.travis-ci.com/IBM/inFairness.svg?branch=main)](https://app.travis-ci.com/IBM/inFairness)
[![License](https://img.shields.io/github/license/ibm/infairness?color=informational)](https://opensource.org/licenses/Apache-2.0)
[![Package versinon](https://img.shields.io/pypi/v/infairness?color=important&label=pypi%20package)](https://pypi.org/project/infairness)
[![Python3.8+](https://img.shields.io/badge/python-3.8+-informational?logo=python)](https://www.python.org/)


----------------


## Installation

inFairness can be installed using `pip`:

```
pip install inFairness
```


Alternatively, if you wish to install the latest development version, you can install directly by cloning the code repository from the GitHub repo:

```
git clone https://github.com/IBM/inFairness
cd inFairness
pip install -e .
```


## Features


inFairness currently supports:

1. Training of individually fair models : [[Docs]](https://ibm.github.io/inFairness/reference/algorithms.html)
2. Auditing pre-trained ML models for individual fairness : [[Docs]](https://ibm.github.io/inFairness/reference/auditors.html)

The package implements the following components:

#### Algorithms

1. Sensitive Set Invariance (SenSeI): [[Paper]](https://arxiv.org/abs/2006.14168), [[Docs]](https://ibm.github.io/inFairness/reference/algorithms.html#sensei-sensitive-set-invariance)
2. Sensitive Subspace Robustness (SenSR): [[Paper]](https://arxiv.org/abs/1907.00020), [[Docs]](https://ibm.github.io/inFairness/reference/algorithms.html#sensr-sensitive-subspace-robustness)

#### Auditors

1. Sensitive Set Invariance (SenSeI) Auditor: [[Paper]](https://arxiv.org/abs/2006.14168), [[Docs]](https://ibm.github.io/inFairness/reference/auditors.html#sensei-auditor)
2. Sensitive Subspace Robustness (SenSR) Auditor: [[Paper]](https://arxiv.org/abs/1907.00020), [[Docs]](https://ibm.github.io/inFairness/reference/auditors.html#sensr-auditor)

#### Metrics

1. Embedded Xenial Pair Logistic Regression Metric (EXPLORE): [[Paper]](https://proceedings.mlr.press/v119/mukherjee20a.html), [[Docs]](https://ibm.github.io/inFairness/reference/distances.html#explore-embedded-xenial-pairs-logistic-regression)
2. SVD Sensitive Subspace Metric: [[Paper]](https://arxiv.org/abs/1907.00020), [[Docs]](https://ibm.github.io/inFairness/reference/distances.html#svd-sensitive-subspace)

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
