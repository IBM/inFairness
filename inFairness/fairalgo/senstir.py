import torch
from torch import nn
from functorch import vmap

from inFairness.auditor import SenSTIRAuditor
from inFairness.distances.mahalanobis_distance import MahalanobisDistances
from inFairness.fairalgo.datainterfaces import FairModelResponse

from inFairness.utils import datautils
from inFairness.utils.plackett_luce import PlackettLuce
from inFairness.utils.ndcg import monte_carlo_vect_ndcg


class SenSTIR(nn.Module):
    """Implementes the Sensitive Subspace Robustness (SenSR) algorithm.

    Proposed in `Individually Fair Ranking <https://arxiv.org/abs/2103.11023>`_

    Parameters
    ------------
        network: torch.nn.Module
            Network architecture
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
        distance_y: inFairness.distances.Distance
            Distance metric in the output space
        rho: float
            :math:`\\rho` parameter in the SenSTIR algorithm (see Algorithm 1)
        eps: float
            :math:`\\epsilon` parameter in the SenSTIR algorithm (see Algorithm 1)
        auditor_nsteps: int
            Number of update steps for the auditor to find worst-case examples
        auditor_lr: float
            Learning rate for the auditor
        monte_carlo_samples_ndcg: int
            Number of monte carlo samples required to estimate the gradient of the
            empirical version of expectation defined in equation SenSTIR in the reference
    """

    def __init__(
        self,
        network: torch.nn.Module,
        distance_x: MahalanobisDistances,
        distance_y: MahalanobisDistances,
        rho: float,
        eps: float,
        auditor_nsteps: int,
        auditor_lr: float,
        monte_carlo_samples_ndcg: int,
    ):
        super().__init__()

        self.network = network
        self.distance_x = distance_x
        self.distance_y = distance_y
        self.rho = rho
        self.eps = eps
        self.auditor_nsteps = auditor_nsteps
        self.auditor_lr = auditor_lr
        self.monte_carlo_samples_ndcg = monte_carlo_samples_ndcg
        self.lamb = None

        self.auditor, self.distance_q = self.__init_auditor__()
        self._vect_gather = vmap(torch.gather, (None, None, 0))

    def __init_auditor__(self):
        auditor = SenSTIRAuditor(
            self.distance_x,
            self.distance_y,
            self.auditor_nsteps,
            self.auditor_lr,
        )

        distance_q = auditor.distance_q
        return auditor, distance_q

    def forward_train(self, Q, relevances):
        batch_size, num_items, num_features = Q.shape
        device = datautils.get_device(Q)

        min_lambda = torch.tensor(1e-5, device=device)

        if self.lamb is None:
            self.lamb = torch.tensor(1.0, device=device)
        if type(self.eps) is float:
            self.eps = torch.tensor(self.eps, device=device)

        if self.rho > 0.0:
            Q_worst = self.auditor.generate_worst_case_examples(
                self.network, Q, self.lamb
            )

            mean_dist_q = self.distance_q(Q, Q_worst).mean()
            # lr_factor = torch.maximum(mean_dist_q, self.eps) / torch.minimum(
            #     mean_dist_q, self.eps
            # )
            lr_factor = 0.5 * self.rho
            self.lamb = torch.maximum(
                min_lambda, self.lamb + lr_factor * (mean_dist_q - self.eps)
            )

            scores = self.network(Q).reshape(batch_size, num_items)  # (B,N,1) --> B,N
            scores_worst = self.network(Q_worst).reshape(batch_size, num_items)

        else:
            scores = self.network(Q).reshape(batch_size, num_items)  # (B,N,1) --> B,N
            scores_worst = torch.ones_like(scores)

        fair_loss = torch.mean(
            -self.__expected_ndcg__(self.monte_carlo_samples_ndcg, scores, relevances)
            + self.rho * self.distance_y(scores, scores_worst)
        )

        response = FairModelResponse(loss=fair_loss, y_pred=scores)
        return response

    def forward_test(self, Q):
        """Forward method during the test phase"""

        scores = self.network(Q).reshape(Q.shape[:2])  # B,N,1 -> B,N
        response = FairModelResponse(y_pred=scores)
        return response

    def forward(self, Q, relevances, **kwargs):
        """Defines the computation performed at every call.

        Parameters
        ------------
            X: torch.Tensor
                Input data
            Y: torch.Tensor
                Expected output data

        Returns
        ----------
            output: torch.Tensor
                Model output
        """

        if self.training:
            return self.forward_train(Q, relevances)
        else:
            return self.forward_test(Q)

    def __expected_ndcg__(self, montecarlo_samples, scores, relevances):
        """
        uses monte carlo samples to estimate the expected normalized discounted cumulative reward
        by using REINFORCE. See section 2 of the reference bellow.

        Parameters
        -------------
            scores: torch.Tensor of dimension B,N
                predicted scores for the objects in a batch of queries

            relevances: torch.Tensor of dimension B,N
                corresponding true relevances of such objects

        Returns
        ------------
            expected_ndcg: torch.Tensor of dimension B
                monte carlo approximation of the expected ndcg by sampling from a Plackett-Luce
                distribution parameterized by :param:`scores`
        """

        prob_dist = PlackettLuce(scores)
        mc_rankings = prob_dist.sample((montecarlo_samples,))
        mc_log_prob = prob_dist.log_prob(mc_rankings)

        mc_relevances = self._vect_gather(relevances, 1, mc_rankings)
        mc_ndcg = monte_carlo_vect_ndcg(mc_relevances)

        expected_utility = (mc_ndcg * mc_log_prob).mean(dim=0)
        return expected_utility
