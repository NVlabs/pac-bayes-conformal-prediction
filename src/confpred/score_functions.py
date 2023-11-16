import torch
from torch import Tensor, nn

from .base import PredictionSet, ScoreFunction
from .pred_sets import BallSet, DiscreteSet, EllipsoidalSet
from .softsort.neural_sort import neural_sort

class ThresholdScoreFunction(ScoreFunction):
    """
    Softmax probability based score function for classification applications
    No learnable parameters
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon  # smoothness parameter

    def forward(self, inputs: Tensor, features: Tensor, outputs: Tensor) -> Tensor:
        # outputs are logits, so turn them into log probabilties
        return torch.log_softmax(outputs, dim=-1)
    
    def nonconformity(self, score_inputs: Tensor, targets: Tensor) -> Tensor:
        # score_inputs is just model outputs with shape [...,J] and represents class probabilities,
        # and targets has shape [...], with values in the range (0,...,J-1)
        pred_probs = torch.gather(score_inputs, -1, targets[..., None])[...,0]
        return -pred_probs

    def level_set(
        self, score_inputs: Tensor, threshold: Tensor
    ) -> PredictionSet:
        scores = -score_inputs
        membership_logits = scores - threshold[...,None]
        membership_scores = torch.sigmoid(-membership_logits / self.epsilon)
        return DiscreteSet(membership_scores=membership_scores)
    
class AdaptiveScoreFunction(ScoreFunction):
    """
    Sum of probabilities before correct label
    No learnable parameters
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon  # smoothness parameter

    def forward(self, inputs: Tensor, features: Tensor, outputs: Tensor) -> Tensor:
        # outputs are logits, so turn them into probabilties
        return torch.softmax(outputs, dim=-1)
    
    def nonconformity(self, score_inputs: Tensor, targets: Tensor) -> Tensor:
        # score_inputs is just model outputs with shape [...,J] and represents class probabilities,
        # and targets has shape [...], with values in the range (0,...,J-1)
        # first, figure out permutation matrix turning probabilities into sorted probabilities
        J = score_inputs.shape[-1]
        permutation = neural_sort(score_inputs, self.epsilon)
        sorted_probs = (permutation @ score_inputs[...,None])[...,0]
        cumulative_dist = torch.cumsum(sorted_probs, -1) - sorted_probs
        scores = (permutation.transpose(-2,-1) @ cumulative_dist[...,None])[...,0]
        scores -= torch.rand_like(scores)*1e-2
        return torch.gather(scores, -1, targets[..., None])[...,0]

    def level_set(
        self, score_inputs: Tensor, threshold: Tensor
    ) -> PredictionSet:
        permutation = neural_sort(score_inputs, self.epsilon)
        sorted_probs = (permutation @ score_inputs[...,None])[...,0]
        cumulative_dist = torch.cumsum(sorted_probs, -1) - sorted_probs
        scores = (permutation.transpose(-2,-1) @ cumulative_dist[...,None])[...,0]
        
        membership_logits = scores - threshold[...,None]
        membership_scores = torch.sigmoid(-membership_logits / self.epsilon)
        return DiscreteSet(membership_scores=membership_scores)

class AdjustedThresholdScoreFunction(ThresholdScoreFunction):
    """
    Applies an input-dependent adjustment to the model's logits
    before computing the score function
    """
    def __init__(self, adjustment_network: nn.Module, epsilon: float = 0.1) -> None:
        super().__init__(epsilon=epsilon)
        self.adjustment_network = adjustment_network
        self._gating_param = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Tensor, features: Tensor,  outputs: Tensor) -> Tensor:
        adjustments = self.adjustment_network(features)
        gate = torch.sigmoid(1e2*self._gating_param)
        adjusted_logits = outputs + gate*adjustments
        
        # outputs are logits, so turn them into log probabilties
        return torch.log_softmax(adjusted_logits, dim=-1)


class EuclideanDistanceScoreFunction(ScoreFunction):
    """
    Simple euclidean distance based score function for regression
    No learnable parameters
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor, features: Tensor, outputs: Tensor) -> Tensor:
        return outputs
    
    def nonconformity(self, score_inputs: Tensor, targets: Tensor) -> Tensor:
        # score_inputs is just model outputs, with shape [...,d] and represents centroids
        # and targets has shape [...,d]
        dist = torch.linalg.norm(score_inputs - targets, dim=-1)
        return dist  # shape [...]

    def level_set(
        self, score_inputs: Tensor, threshold: Tensor
    ) -> PredictionSet:
        return BallSet(centroid=score_inputs, radius=threshold)

class HeteroskedasticScoreFunction(ScoreFunction):
    """
    Trainable input-dependent uncertainty scaled distance for regression
    """

    def __init__(self, unc_network: nn.Module, min_unc=1e-2) -> None:
        super().__init__()
        self._unc_network = unc_network
        self._default_unc = 1.
        self._gating_param = nn.Parameter(torch.zeros(1))
        self._min_unc = 1e-2

    def _unc(self, inputs: Tensor) -> Tensor:
        unc = self._default_unc
        correction = -1 + torch.nn.functional.softplus(self._unc_network(inputs)[...,0] + 0.6)
        gate = torch.sigmoid(2e3*self._gating_param) # TODO: make this work for all prior_scales
        return unc + gate*correction + self._min_unc

    def forward(self, inputs: Tensor, features: Tensor, outputs: Tensor) -> Tensor:
        # use inputs to estimate input-dependant uncertainty scaling factor
        unc = self._unc(inputs)
        # tack it on to outputs
        return torch.cat([outputs, unc[...,None]], dim=-1)
    
    def nonconformity(self, score_inputs: Tensor, targets: Tensor) -> Tensor:
        # score inputs outputs with unc added on to last dim
        outputs = score_inputs[...,:-1]
        unc = score_inputs[...,-1]
        dist = torch.linalg.norm(outputs - targets, dim=-1)
        return dist / unc # shape [...]

    def level_set(
        self, score_inputs: Tensor, threshold: Tensor
    ) -> PredictionSet:
        outputs = score_inputs[...,:-1]
        unc = score_inputs[...,-1]
        return BallSet(centroid=outputs, radius=threshold * unc)
