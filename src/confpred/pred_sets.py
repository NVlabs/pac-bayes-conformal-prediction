from dataclasses import dataclass
from typing import List

import torch
from torch import BoolTensor, LongTensor, Tensor

from .base import PredictionSet


@dataclass
class DiscreteSet(PredictionSet):
    """
    Represents a discrete subset of J elements
    by storing membership scores for each of the J elements
    membership scores are stored as floats to allow for soft membership,
    and thresholded at 0.5 to provide a hard cutoff on set membership
    """

    membership_scores: Tensor  # [..., J]

    def volume(self) -> Tensor:
        return (self.membership_scores > 0.5).float().sum(-1)

    def differentiable_volume(self) -> Tensor:
        # stop gradients if set size dips below 1
        return torch.clamp(self.membership_scores.sum(-1) - 1, min=0) + 1

    def in_set(self, element: LongTensor) -> BoolTensor:
        # here, element is a set of ints, shape [...], with each element in {0,...,J-1}
        element = element.expand(self.membership_scores[...,0].shape)
        relevant_scores = torch.gather(self.membership_scores, -1, element[...,None])[...,0]
        return relevant_scores > 0.5


@dataclass
class EllipsoidalSet(PredictionSet):
    """
    Represents the ellipse defined by
     (x - centroid).T @ precision_matrix @ (x - centroid) <= 1
    """

    centroid: Tensor  # [..., d]
    precision_matrix: Tensor  # [..., d, d]

    def volume(self) -> Tensor:
        return torch.exp( -torch.logdet(self.precision_matrix) )

    def in_set(self, element: Tensor) -> BoolTensor:
        # Here, element must be  [..., d]
        diff = element - self.centroid
        transformed_sq_dist = torch.einsum(
            "...ij,...i,...j", self.precision_matrix, diff, diff
        )
        return transformed_sq_dist <= 1


@dataclass
class BallSet(PredictionSet):
    """
    Represents a ball with a given radius around a centroid
    """

    centroid: Tensor  # [..., d]
    radius: Tensor  # [...]

    def volume(self) -> Tensor:
        return self.centroid.shape[-1] * torch.log(self.radius)

    def in_set(self, element: Tensor) -> BoolTensor:
        # Here, element must be  [..., d]
        diff = element - self.centroid
        return torch.linalg.norm(diff, dim=-1) <= self.radius
