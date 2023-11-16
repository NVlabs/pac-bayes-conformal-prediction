from typing import Optional
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from ..base import ConfPredictor, PredictionSet, ScoreFunction
from ..softsort.neural_sort import soft_quantile


class StandardConfPred(ConfPredictor):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        score_function: ScoreFunction,
        alpha: float,
        delta: float,
        alpha_hat: float,
        optimize_model: bool = False,
        optimize_score_function: bool = True,
        **kwargs
    ):
        super().__init__(
            backbone,
            head,
            score_function,
            alpha,
            delta,
            alpha_hat,
            optimize_model,
            optimize_score_function,
        )

        self.has_prior_loss = True
        self.has_prior_constraint = False

        self.register_buffer(
            "threshold", torch.full((1,), torch.nan, dtype=torch.float)
        )

    def prior_loss(self, inputs: Tensor, targets: Tensor, N: int) -> Tensor:
        n = inputs.shape[0]
        score_inputs = self.score_inputs(inputs)
        scores = self.score_function.nonconformity(score_inputs, targets)
        threshold = soft_quantile(scores, self._q(n, self.alpha_hat))
        pred_set = self.score_function.level_set(score_inputs, threshold)

        # loss is mean size of prediction sets
        return pred_set.differentiable_volume().mean()

    def calibrate(self, dataset: Dataset, **dl_kwargs: Optional[dict]) -> None:
        N = len(dataset)
        if "batch_size" in dl_kwargs.keys():
            dl_kwargs['batch_size'] = N
        
        dataloader = DataLoader(dataset=dataset, **dl_kwargs)
        
        all_scores = []
        for inputs,targets in dataloader:
            inputs = inputs.to(self.threshold.device)
            targets = targets.to(self.threshold.device)
            with torch.no_grad():
                all_scores.append( self.scores(inputs, targets).detach() )
        
        scores = torch.concat(all_scores, dim=0)
        self.threshold.data = torch.quantile(
            scores, self._q(N, self.alpha_hat)
        ).reshape(1)

    def forward(
        self, inputs: Tensor, score_quantile: Optional[Tensor] = None
    ) -> PredictionSet:
        if score_quantile is None:
            if torch.isnan(self.threshold):
                raise ValueError("Model not calibrated! Must call self.calibrate first")

            score_quantile = self.threshold

        score_inputs = self.score_inputs(inputs)
        pred_set = self.score_function.level_set(score_inputs, score_quantile)
        return pred_set
