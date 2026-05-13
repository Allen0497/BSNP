"""
Evaluation metrics used in paper Section 5.2.

  MNSE  (Mean Normalized Squared Error) - pointwise accuracy
  NLL   (Negative Log-Likelihood)       - probabilistic quality
  ECP   (Empirical Coverage Probability) - uncertainty calibration (at a nominal level)
"""
import torch
from scipy import stats


def mnse(mu_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """MNSE = || mu - y ||^2 / || y ||^2."""
    return ((mu_pred - y_true) ** 2).sum() / (y_true ** 2).sum()


def nll(mu_pred: torch.Tensor, sigma_pred: torch.Tensor,
        y_true: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL under the predictive distribution."""
    nll_val = 0.5 * torch.log(torch.tensor(2 * torch.pi)) \
              + sigma_pred.log() \
              + 0.5 * ((y_true - mu_pred) / sigma_pred) ** 2
    return nll_val.mean()


def ecp(mu_pred: torch.Tensor, sigma_pred: torch.Tensor,
        y_true: torch.Tensor, level: float = 0.9) -> float:
    """
    Empirical Coverage Probability of the central (1 - alpha) predictive interval.
    A well-calibrated model should produce ecp ≈ level.
    """
    z = stats.norm.ppf(0.5 + level / 2)
    lower = mu_pred - z * sigma_pred
    upper = mu_pred + z * sigma_pred
    covered = ((y_true >= lower) & (y_true <= upper)).float()
    return covered.mean().item()
