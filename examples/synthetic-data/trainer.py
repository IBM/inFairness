class Trainer(object):
    """Main trainer class that orchestrates the entire learning routine
    Use this class to start training a model using individual fairness routines

    Args:
        dataloader (torch.util.data.DataLoader): training data loader
        model (inFairness.fairalgo): Individual fairness algorithm
        optimizer (torch.optim): Model optimizer
        max_iterations (int): Number of training steps
    """

    def __init__(self, dataloader, model, optimizer, max_iterations):

        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.max_iterations = max_iterations

        self._dataloader_iter = iter(self.dataloader)

    def run_step(self):

        try:
            data = next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self.dataloader)
            data = next(self._dataloader_iter)

        if isinstance(data, list) or isinstance(data, tuple):
            model_output = self.model(*data)
        elif isinstance(data, dict):
            model_output = self.model(**data)
        else:
            raise AttributeError(
                "Data format not recognized. Only `list`, `tuple`, and `dict` are recognized."
            )

        self.optimizer.zero_grad()
        model_output.loss.backward()

        self.optimizer.step()

    def train(self):

        self.model.train(True)

        for step_count in range(self.max_iterations):
            self.run_step()
