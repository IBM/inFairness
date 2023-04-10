### inFairness

<br />

<p align="center">
   <a href="https://pypi.org/project/infairness"><img src="https://img.shields.io/pypi/v/infairness?color=important&label=pypi%20package"></a>
   <a href="https://github.com/IBM/inFairness"><img src="https://img.shields.io/badge/code-GitHub-informational?logo=github"></a>
   <a href="https://github.com/IBM/inFairness/tree/main/examples"><img src="https://img.shields.io/badge/example-notebooks-red?logo=jupyter"></a>
   <a href="https://ibm.github.io/inFairness"><img src="https://img.shields.io/badge/documentation-up-green?logo=GitBook"></a>
   <a href="https://fairbert.vizhub.ai"><img src="https://img.shields.io/badge/fairness-demonstration-yellow?logo=ibm-watson"></a>
   <br/>
   <a href="https://app.travis-ci.com/IBM/inFairness"><img src="https://app.travis-ci.com/IBM/inFairness.svg?branch=main"></a>
   <a href="https://pypistats.org/packages/infairness"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/inFairness?color=blue"></a>
   <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.7+-blue?logo=python"></a>
   <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/license-Apache-yellow"></a>
   <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<br />

Intuitively, an individually fair Machine Learning (ML) model treats similar inputs similarly. Formally, the leading notion of individual fairness is metric fairness [(Dwork et al., 2011)](https://dl.acm.org/doi/abs/10.1145/2090236.2090255); it requires:

$$ d_y (h(x_1), h(x_2)) \leq L d_x(x_1, x_2) \quad \forall \quad x_1, x_2 \in X $$

Here, $h: X \rightarrow Y$ is a ML model, where $X$ and $Y$ are input and output spaces; $d_x$ and $d_y$ are metrics on the input and output spaces, and $L \geq 0$ is a Lipchitz constant. This constrained optimization equation states that the distance between the model predictions for inputs $x_1$ and $x_2$ is upper-bounded by the fair distance between the inputs $x_1$ and $x_2$. Here, the fair metric $d_x$ encodes our intuition of which samples should be treated similarly by the ML model, and in designing so, we ensure that for input samples considered similar by the fair metric $d_x$, the model outputs will be similar as well.

inFairness is a PyTorch package that supports auditing, training, and post-processing ML models for individual fairness. At its core, the library implements the key components of individual fairness pipeline: $d_x$ - distance in the input space, $d_y$ - distance in the output space, and the learning algorithms to optimize for the equation above.

For an in-depth tutorial of Individual Fairness and the inFairness package, please watch this tutorial. Also, take a look at the [examples](https://github.com/IBM/inFairness/tree/main/examples) folder for illustrative use-cases.

<p align="center">
<iframe width="480" height="270" src="https://video.ibm.com/embed/recorded/131932983" scrolling="no" allowfullscreen webkitallowfullscreen frameborder="0" style="border: 0 none transparent;"></iframe>
</p>


----------------


### Installation

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


### Features


inFairness currently supports:

1. Training of individually fair models : [[Docs]](https://ibm.github.io/inFairness/reference/algorithms.html)
2. Auditing pre-trained ML models for individual fairness : [[Docs]](https://ibm.github.io/inFairness/reference/auditors.html)
3. Post-processing for Individual Fairness : [[Docs]](https://ibm.github.io/inFairness/reference/postprocessing.html)

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

#### Post-Processing algorithms

1. Graph Laplacian Individual Fairness (GLIF): [[Paper]](https://arxiv.org/abs/2110.13796), [[Docs]](https://ibm.github.io/inFairness/reference/postprocessing.html#graph-laplacian-individual-fairness-glif)

----------

### API Documentation

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
