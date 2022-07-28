# Examples

#### Auditing models for Individual Fairness
| Task      | Auditor | Fair Metrics |  |
| ----------- | ----------- | ----------- | ----------- |
| Adult Income Prediction      | [SenSeI](https://ibm.github.io/inFairness/reference/auditors.html#sensei-auditor), [SenSR](https://ibm.github.io/inFairness/reference/auditors.html#sensr-auditor) |  [LogisticRegSensitiveSubspace](https://ibm.github.io/inFairness/reference/distances.html#logistic-regression-sensitive-subspace-distance-metric) |  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ibm/infairness/main?labpath=examples%2Fadult-income-prediction%2Fadult_income_prediction.ipynb)     |
| Sentiment Analysis   | [SenSR](https://ibm.github.io/inFairness/reference/auditors.html#sensr-auditor) | [SVDSensitiveSubspaceDistance](https://ibm.github.io/inFairness/reference/distances.html#svd-sensitive-subspace) |  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ibm/infairness/main?labpath=examples%2Fsentiment-analysis%2Fsentiment_analysis_demo.ipynb)    |

-------

#### Training models for Individual Fairness
| Task      | FairAlgo | Fair Metrics |  |
| ----------- | ----------- | ----------- | ----------- |
| Adult Income Prediction      | [SenSeI](https://ibm.github.io/inFairness/reference/algorithms.html#sensei-sensitive-set-invariance) |  [LogisticRegSensitiveSubspace](https://ibm.github.io/inFairness/reference/distances.html#logistic-regression-sensitive-subspace-distance-metric) |  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ibm/infairness/main?labpath=examples%2Fadult-income-prediction%2Fadult_income_prediction.ipynb)   |
| Sentiment Analysis   | [SenSeI](https://ibm.github.io/inFairness/reference/algorithms.html#sensei-sensitive-set-invariance) | [SVDSensitiveSubspaceDistance](https://ibm.github.io/inFairness/reference/distances.html#svd-sensitive-subspace) |  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ibm/infairness/main?labpath=examples%2Fsentiment-analysis%2Fsentiment_analysis_demo.ipynb)    |
| Synthetic Data   | [SenSeI](https://ibm.github.io/inFairness/reference/algorithms.html#sensei-sensitive-set-invariance) | [ProtectedEuclidenDistance](https://ibm.github.io/inFairness/reference/distances.html#protected-euclidean-distance) |  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ibm/infairness/main?labpath=examples%2Fsynthetic-data%2Fsynthetic_data_demo.ipynb)    |
| Word Embedding Association Tests | --- |  [EXPLOREDistance](https://ibm.github.io/inFairness/reference/distances.html#explore-embedded-xenial-pairs-logistic-regression)  |  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ibm/infairness/main?labpath=examples%2Fword-embedding-association-test%2Fweat-explore.ipynb)      |

--------

#### Post-processing pre-trained model predictions for Individual Fairness
| Task      | Fair Metrics | Link |
| ----------- | ----------- | ----------- |
| Sentiment Analysis (DistilBERT model from HuggingFace)      | [SVDSensitiveSubspaceDistance](https://ibm.github.io/inFairness/reference/distances.html#svd-sensitive-subspace) | [![Binder](https://mybinder.org/badge_logo.svg)](ttps://mybinder.org/v2/gh/ibm/infairness/main?labpath=examples%2Fpostprocess-sentiment-analysis%2Fpostprocess.ipynb)       |
