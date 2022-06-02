# Examples


Synthetic data
-------------------

```{eval-rst}

.. list-table::

    * - In this experiment, we generate random 2-dimensional data, and train a classifier
        with one dimension explicitly specified as a protected attribute. In the figure here,
        we x-axis is defined to be a protected attribute, thus, the classifier learns to ignore
        the protected attribute while making predictions.

        .. button-link:: https://github.com/IBM/inFairness/tree/main/examples/synthetic-data
            :expand:
            :color: secondary

            Experiment Link

      - .. figure:: _static/imgs/example-synthetic-data.png

```

-------------------

Individually fair sentiment classifier
-------------------

```{eval-rst}

.. list-table::

    * - Ideally, sentiment classifiers should not assign a higher sentiment to
        names predominant of one particular community over another. Unfortunately,
        sentiment classifiers trained on standard word embeddings do assign a higher 
        sentiment to predominantly white names as opposed to predominantly black names.
        
        In this example, we first show that standard sentiment classifier exhibit this
        undesirable property. Thereafter, we train an individually fair sentiment classifier
        that assigns similar sentiments to names from the two communities.

        .. button-link:: https://github.com/IBM/inFairness/tree/main/examples/sentiment-analysis
            :expand:
            :color: secondary

            Experiment Link

      - .. figure:: _static/imgs/example-sentiment-clf.png

```

-------------------

Word Embedding Association Tests
-------------------

```{eval-rst}

.. list-table::

    * - Many recent works have observed biases in word embeddings. `Caliskan et al. (2017) <https://arxiv.org/pdf/1608.07187.pdf>`_
        proposed a methodological way of analyzing various biases through a series of Word Embedding Association Tests (WEATs). 
        In this experiment, we show that replacing the metric on the word embedding space with a fair metric such as EXPLORE 
        eliminates most biases in word embeddings.

        .. button-link:: https://github.com/IBM/inFairness/tree/main/examples/word-embedding-association-test
            :expand:
            :color: secondary

            Experiment Link

      - .. figure:: _static/imgs/example-weat.png

```
