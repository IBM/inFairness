import torch

from inFairness.distances.distance import Distance


class EuclideanDistance(Distance):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, itemwise_dist=True):

        if itemwise_dist:
            return torch.cdist(x.unsqueeze(1), y.unsqueeze(1)).reshape(-1, 1)
        else:
            return torch.cdist(x, y)


class ProtectedEuclideanDistance(Distance):
    def __init__(self):
        super().__init__()

        self._protected_attributes = None
        self._num_attributes = None
        self.protected_vector = None

    def to(self, device):
        """Moves distance metric to a particular device

        Parameters
        ------------
            device: torch.device
        """

        assert (
            self.protected_vector is not None
        ), "Please fit the metric before moving parameters to device"

        self.device = device
        self.protected_vector = self.protected_vector.to(self.device)

    def fit(self, protected_attributes, num_attributes):
        """Fit Protected Euclidean Distance metric

        Parameters
        ------------
            protected_attributes: Iterable[int]
                List of attribute indices considered to be protected.
                The metric would ignore these protected attributes while
                computing distance between data points.
            num_attributes: int
                Total number of attributes in the data points.
        """

        self._protected_attributes = protected_attributes
        self._num_attributes = num_attributes

        self.protected_vector = torch.ones(num_attributes)
        self.protected_vector[protected_attributes] = 0.0

    def forward(self, x, y, itemwise_dist=True):
        """
        :param x, y: a B x D matrices
        :return: B x 1 matrix with the protected distance camputed between x and y
        """
        protected_x = (x * self.protected_vector).unsqueeze(1)
        protected_y = (y * self.protected_vector).unsqueeze(1)

        if itemwise_dist:
            return torch.cdist(protected_x, protected_y).reshape(-1, 1)
        else:
            return torch.cdist(protected_x, protected_y)
