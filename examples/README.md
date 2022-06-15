# Examples

> **Warning**
> Example notebooks will currently fail to execute in Google Colaboratory. This is because Colab provides Python 3.7, while `inFairness` package requires Python 3.8 and above. The Colab team is working on upgrading the Colab environment to Python 3.8 and it is being tracked in the bug [here](https://github.com/googlecolab/colabtools/issues/1880). Users should be able to execute the notebooks on their local systems and on Colab once the environment update is complete.

#### Auditing models for Individual Fairness
| Task      | Auditor | Fair Metrics |  |
| ----------- | ----------- | ----------- | ----------- |
| Adult Income Prediction      | [SenSeI](https://ibm.github.io/inFairness/reference/auditors.html#sensei-auditor), [SenSR](https://ibm.github.io/inFairness/reference/auditors.html#sensr-auditor) |  [LogisticRegSensitiveSubspace](https://ibm.github.io/inFairness/reference/distances.html#logistic-regression-sensitive-subspace-distance-metric) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/IBM/inFairness/blob/main/examples/adult-income-prediction/adult_income_prediction.ipynb)     |
| Sentiment Analysis   | [SenSR](https://ibm.github.io/inFairness/reference/auditors.html#sensr-auditor) | [SVDSensitiveSubspaceDistance](https://ibm.github.io/inFairness/reference/distances.html#svd-sensitive-subspace) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/IBM/inFairness/blob/main/examples/sentiment-analysis/sentiment_analysis_demo.ipynb)    |

-------

#### Training models for Individual Fairness
| Task      | FairAlgo | Fair Metrics |  |
| ----------- | ----------- | ----------- | ----------- |
| Adult Income Prediction      | [SenSeI](https://ibm.github.io/inFairness/reference/algorithms.html#sensei-sensitive-set-invariance) |  [LogisticRegSensitiveSubspace](https://ibm.github.io/inFairness/reference/distances.html#logistic-regression-sensitive-subspace-distance-metric) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/IBM/inFairness/blob/main/examples/adult-income-prediction/adult_income_prediction.ipynb)    |
| Sentiment Analysis   | [SenSeI](https://ibm.github.io/inFairness/reference/algorithms.html#sensei-sensitive-set-invariance) | [SVDSensitiveSubspaceDistance](https://ibm.github.io/inFairness/reference/distances.html#svd-sensitive-subspace) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/IBM/inFairness/blob/main/examples/sentiment-analysis/sentiment_analysis_demo.ipynb)    |
| Synthetic Data   | [SenSeI](https://ibm.github.io/inFairness/reference/algorithms.html#sensei-sensitive-set-invariance) | [ProtectedEuclidenDistance](https://ibm.github.io/inFairness/reference/distances.html#protected-euclidean-distance) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/IBM/inFairness/blob/main/examples/synthetic-data/synthetic_data_demo.ipynb)      |
| Word Embedding Association Tests | --- |  [EXPLOREDistance](https://ibm.github.io/inFairness/reference/distances.html#explore-embedded-xenial-pairs-logistic-regression)  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/IBM/inFairness/blob/main/examples/word-embedding-association-test/weat-explore.ipynb)      |
