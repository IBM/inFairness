import torch
from torch.nn.parameter import Parameter

from inFairness.distances import (
    WassersteinDistance,
    MahalanobisDistances,
)
from inFairness.auditor import Auditor

from inFairness.utils.params import freeze_network, unfreeze_network


class SenSTIRAuditor(Auditor):
    """SenSTIR Auditor generates worst-case examples by solving the
    following optimization problem:

    .. math:: q^{'} \gets arg\max_{q^{'}}\{||h_{\\theta_t}(q),h_{\\theta_t}(q^{'})||_{2}^{2} - \lambda_t d_{Q}(q,q^{'})\}

    At a high level, it will find :math:`q^{'}` such that it maximizes the score difference, while keeping
    a fair set distance `distance_q` with the original query `q` small.

    Proposed in `Individually Fair Rankings <https://arxiv.org/abs/2103.11023>`_


    Parameters
    -----------
      distance_x: inFairness.distances.Distance
        Distance metric in the input space. Should be an instance of
        :class:`~inFairness.distances.MahalanobisDistance`
      distance_y: inFairness.distances.Distance
        Distance metric in the output space. Should be an instance of
        :class:`~inFairness.distances.MahalanobisDistance`
      num_steps: int 
        number of optimization steps taken to produce the worst examples.
      lr: float
        learning rate of the optimization
      max_noise: float 
        range of a uniform distribution determining the initial noise added to q to form q'
      min_noise: float
        range of a uniform distribution determining the initial noise added to q to form q'
    """

    def __init__(
        self,
        distance_x: MahalanobisDistances,
        distance_y: MahalanobisDistances,
        num_steps: int,
        lr: float,
        max_noise: float = 0.1,
        min_noise: float = -0.1,
    ):
        self.distance_x = distance_x
        self.distance_y = distance_y
        self.num_steps = num_steps
        self.lr = lr
        self.max_noise = max_noise
        self.min_noise = min_noise

        self.distance_q = self.__init_query_distance__()

    def __init_query_distance__(self):
        """Initialize Wasserstein distance metric from provided input distance metric"""

        sigma_ = self.distance_x.sigma
        distance_q = WassersteinDistance()
        distance_q.fit(sigma=sigma_)
        return distance_q

    def generate_worst_case_examples(self, network, Q, lambda_param, optimizer=None):
        """Generate worst case examples given the input sample batch of queries Q (dimensions batch_size,num_items,num_features)

        Parameters
        -----------
          network: torch.nn.Module
            PyTorch network model that outputs scores per item
          Q: torch.Tensor
            tensor with dimensions batch_size, num_items, num_features containing the batch of queries for ranking
          lambda_param: torch.float
            Lambda weighting parameter as defined above
          optimizer: torch.optim.Optimizer, optional
            Pytorch Optimizer object

        Returns
        ---------
          q_worst: torch.Tensor
            worst case queries for the provided input queries `Q`
        """
        assert optimizer is None or issubclass(optimizer, torch.optim.Optimizer)

        batch_size, num_items, _ = Q.shape
        freeze_network(network)
        lambda_param = lambda_param.detach()

        delta = Parameter(
            torch.rand_like(Q) * (self.max_noise - self.min_noise) + self.min_noise
        )

        if optimizer is None:
            optimizer = torch.optim.Adam([delta], lr=self.lr)
        else:
            optimizer = optimizer([delta], lr=self.lr)

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            Q_worst = Q + delta
            input_dist = self.distance_q(Q, Q_worst)  # this is of size B

            out_Q = network(Q).reshape(
                batch_size, num_items
            )  # shape B,N,1 scores --> B,N
            out_Q_worst = network(Q_worst).reshape(batch_size, num_items)

            out_dist = self.distance_y(out_Q, out_Q_worst)
            out_dist = out_dist.reshape(
                -1
            )  # distance_y outputs B,1 whereas input_dist is B.

            loss = (-(out_dist - lambda_param * input_dist)).sum()
            loss.backward()
            optimizer.step()

        unfreeze_network(network)

        return (Q + delta).detach()
