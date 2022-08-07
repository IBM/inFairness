<p align="center">
  <a href="https://ibm.github.io/inFairness">
     <img width="350" height="350" src="https://ibm.github.io/inFairness/_static/infairness-logo.png">
   </a>
</p>

<p align="center">
   <a href="https://pypi.org/project/infairness"><img src="https://img.shields.io/pypi/v/infairness?color=important&label=pypi%20package&logo=PyPy"></a>
   <a href="./examples"><img src="https://img.shields.io/badge/example-notebooks-red?logo=jupyter"></a>
   <a href="https://ibm.github.io/inFairness"><img src="https://img.shields.io/badge/documentation-up-green?logo=GitBook"></a>
   <br/>
   <a href="https://app.travis-ci.com/IBM/inFairness"><img src="https://app.travis-ci.com/IBM/inFairness.svg?branch=main"></a>
   <a href="https://pepy.tech/project/infairness"><img src="https://pepy.tech/badge/infairness"></a>
   <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8+-blue?logo=python"></a>
   <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/license-Apache-yellow"></a>
   <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


## Individual Fairness and inFairness

Intuitively, an individually fair Machine Learning (ML) model treats similar inputs similarly. Formally, the leading notion of individual fairness is metric fairness [(Dwork et al., 2011)](https://dl.acm.org/doi/abs/10.1145/2090236.2090255); it requires:

$$ d_y (h(x_1), h(x_2)) \leq L d_x(x_1, x_2) \quad \forall \quad x_1, x_2 \in X $$

Here, $h: X \rightarrow Y$ is a ML model, where $X$ and $Y$ are input and output spaces; $d_x$ and $d_y$ are metrics on the input and output spaces, and $L \geq 0$ is a Lipchitz constant. This constrained optimization equation states that the distance between the model predictions for inputs $x_1$ and $x_2$ is upper-bounded by the fair distance between the inputs $x_1$ and $x_2$. Here, the fair metric $d_x$ encodes our intuition of which samples should be treated similarly by the ML model, and in designing so, we ensure that for input samples considered similar by the fair metric $d_x$, the model outputs will be similar as well.

inFairness is a PyTorch package that supports auditing, training, and post-processing ML models for individual fairness. At its core, the library implements the key components of individual fairness pipeline: $d_x$ - distance in the input space, $d_y$ - distance in the output space, and the learning algorithms to optimize for the equation above.

For an in-depth tutorial of Individual Fairness and the inFairness package, please watch this tutorial. Also, take a look at the [examples](./examples/) folder for illustrative use-cases. For more group fairness examples see [AIF360](https://aif360.mybluemix.net/).

<p align="center">
  <a href="https://video.ibm.com/recorded/131932983" target="_blank"><img width="700" alt="Watch the tutorial" src="https://user-images.githubusercontent.com/991913/178768336-2bfa5958-487f-4f14-a156-03dacfd68263.png"></a>
</p>

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

1. Learning individually fair metrics : [[Docs]](https://ibm.github.io/inFairness/reference/distances.html)
2. Training of individually fair models : [[Docs]](https://ibm.github.io/inFairness/reference/algorithms.html)
3. Auditing pre-trained ML models for individual fairness : [[Docs]](https://ibm.github.io/inFairness/reference/auditors.html)
4. Post-processing for Individual Fairness : [[Docs]](https://ibm.github.io/inFairness/reference/postprocessing.html)


### Coming soon

We plan to extend the package by integrating the following features:
1. Individually fair ranking : [[Paper]](https://arxiv.org/abs/2103.11023)


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

<table align="center">
  <tr>
    <td align="center"><a href="http://moonfolk.github.io/"><img src="https://avatars.githubusercontent.com/u/24443134?v=4?s=100" width="120px;" alt=""/><br /><b>Mikhail Yurochkin</b></a></a></td>
    <td align="center"><a href="http://mayankagarwal.github.io/"><img src="https://avatars.githubusercontent.com/u/991913?v=4?s=100" width="120px;" alt=""/><br /><b>Mayank Agarwal</b></a></a></td>
    <td align="center"><a href="https://github.com/aldopareja"><img src="https://avatars.githubusercontent.com/u/7622817?v=4?s=100" width="120px;" alt=""/><br /><b>Aldo Pareja</b></a></a></td>
    <td align="center"><a href="https://github.com/onkarbhardwaj"><img src="https://avatars.githubusercontent.com/u/13560220?v=4?s=100" width="120px;" alt=""/><br /><b>Onkar Bhardwaj</b></a></a></td>
  </tr>
</table>
