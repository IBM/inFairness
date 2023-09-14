# Tutorial


The [`examples`](https://github.com/IBM/inFairness/tree/main/examples/) folder contains tutorials from different fields illustrating how to use the package.

## Minimal example

First, you need to import the relevant packages

```
from inFairness.distances import SVDSensitiveSubspaceDistance, EuclideanDistance
from inFairness.fairalgo import SenSeI
```

The `inFairness.distances` module implements various distance metrics on the input and the output spaces, and the `inFairness.fairalgo` implements various individually fair learning algorithms with `SenSeI` being one particular algorithm.

Thereafter, we instantiate and fit the distance metrics on the training data, and 


```
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
