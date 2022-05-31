import torch
import numpy as np

from abc import ABCMeta
from scipy.stats import norm
from inFairness.utils.datautils import convert_tensor_to_numpy
from inFairness.auditor.datainterface import AuditorResponse


class Auditor(metaclass=ABCMeta):
    """
    Abstract class for model auditors, e.g. Sensei or Sensr
    """

    def __init__(self):
        super(Auditor, self).__init__()

    def generate_worst_case_examples(self, *args, **kwargs):
        """Generates worst-case example for the input data sample batch"""
        raise NotImplementedError(
            "Method `generate_worst_case_examples` not implemented."
        )

    def compute_loss_ratio(self, X_audit, X_worst, Y_audit, network, loss_fn):
        """Compute the loss ratio of samples computed by solving gradient flow attack
        to original audit samples

        Parameters
        -------------
            X_audit: torch.Tensor
                Auditing samples. Shape (n_samples, n_features)
            Y_audit: torch.Tensor
                Labels of auditing samples. Shape: (n_samples)
            lambda_param: float
                Lambda weighting parameter as defined in the equation above

        Returns
        ---------
            loss_ratios: numpy.ndarray
                Ratio of loss for samples computed using gradient
                flow attack to original audit samples
        """

        with torch.no_grad():
            Y_pred_worst = network(X_worst)
            Y_pred_original = network(X_audit)

            loss_vals_adversarial = loss_fn(Y_pred_worst, Y_audit, reduction="none")
            loss_vals_original = loss_fn(Y_pred_original, Y_audit, reduction="none")

        loss_vals_adversarial = convert_tensor_to_numpy(loss_vals_adversarial)
        loss_vals_original = convert_tensor_to_numpy(loss_vals_original)

        loss_ratio = np.divide(loss_vals_adversarial, loss_vals_original)

        return loss_ratio

    def compute_audit_result(self, loss_ratios, threshold=None, confidence=0.95):
        """Computes auditing statistics given loss ratios and user-specified
        acceptance threshold

        Parameters
        -------------
            loss_ratios: numpy.ndarray
                List of loss ratios between worst-case and normal data samples
            threshold: float. optional
                User-specified acceptance threshold value
                If a value is not specified, the procedure simply returns the mean
                and lower bound of loss ratio, leaving the detemination of models'
                fairness to the user.
                If a value is specified, the procedure also determines if the model
                is individually fair or not.
            confidence: float, optional
                Confidence value. Default = 0.95

        Returns
        ----------
            audit_result: AuditorResponse
                Data interface with auditing results and statistics
        """

        loss_ratios = loss_ratios[np.isfinite(loss_ratios)]

        lossratio_mean = np.mean(loss_ratios)
        lossratio_std = np.std(loss_ratios)
        N = loss_ratios.shape[0]

        z = norm.ppf(confidence)
        lower_bound = lossratio_mean - z * lossratio_std / np.sqrt(N)

        if threshold is None:
            response = AuditorResponse(
                lossratio_mean=lossratio_mean,
                lossratio_std=lossratio_std,
                lower_bound=lower_bound,
            )
        else:
            tval = (lossratio_mean - threshold) / lossratio_std
            tval *= np.sqrt(N)

            pval = 1 - norm.cdf(tval)
            is_model_fair = False if pval < (1 - confidence) else True

            response = AuditorResponse(
                lossratio_mean=lossratio_mean,
                lossratio_std=lossratio_std,
                lower_bound=lower_bound,
                threshold=threshold,
                pval=pval,
                confidence=confidence,
                is_model_fair=is_model_fair,
            )

        return response

    def audit(self, *args, **kwargs):
        """Audit model for individual fairness"""
        raise NotImplementedError("Method not implemented")
