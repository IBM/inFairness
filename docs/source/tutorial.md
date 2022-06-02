# Tutorial


The [`examples`](https://github.com/IBM/inFairness/tree/main/examples/) folder contains tutorials from different fields illustrating how to use the package.

Some of these examples include:
1. Individually fair sentiment classifier: [Notebook](https://github.com/IBM/inFairness/tree/main/examples/sentiment-analysis/sentiment_analysis_demo_api.ipynb)

## Minimal example

First, you need to import the relevant packages

```
from inFairness.auditor import SenSeIAuditor
from inFairness.distances import SVDSensitiveSubspaceDistance, EuclideanDistance
from inFairness.fairalgo import SenSeI
```

The `inFairness.auditor.SenSeIAuditor` helps the fairness algorithm compute the adversarial example given the input, `inFairness.distances` implement various distance metrics on the input and the output spaces, and the `inFairness.fairalgo.SenSeI` is the individually fair algorithm that ties all the components together.

Thereafter, we initialize the auditor, specify the distances, and 


```
auditor = SenSeIAuditor(num_steps=100, lr=0.1)
distance_x = SVDSensitiveSubspaceDistance(X_train=data, n_components=50)
distance_y = EuclideanDistance()

# Finally instantiate the fair algorithm
fairalgo = SenSeI(auditor, distance_x, distance_y, network, lossfn, rho=1.0, eps=1e-3, lr=0.01)
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
