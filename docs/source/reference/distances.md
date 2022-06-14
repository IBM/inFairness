# Distances

Consider an ML model as a map $h: X \rightarrow Y$, where $(X, d_x)$ and 
$(Y, d_y)$ are the input and output metric spaces respectively. Individual 
fairness is $L$-Lipschitz continuity of $h$:

$$ d_y(h(x_1), h(x_2)) \leq L d_x(x_1, x_2) \text{ for all } x_1, x_2 \in X $$

This sub-module contains some commonly defined metrics in the input ($d_x$) and output ($d_y$) spaces

```{eval-rst}

.. currentmodule:: inFairness.distances

Mahalanobis distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: inFairness.distances.MahalanobisDistances
    :members:

Sensitive Subspace distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: inFairness.distances.SensitiveSubspaceDistance
    :members:

SVD Sensitive Subspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SVDSensitiveSubspaceDistance
    :members:


EXPLORE: Embedded Xenial Pairs Logistic Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: EXPLOREDistance
    :members:

Logistic Regression Sensitive Subspace distance metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LogisticRegSensitiveSubspace
    :members:

Protected Euclidean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ProtectedEuclideanDistance
    :members:

Euclidean Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: EuclideanDistance
    :members:
```
