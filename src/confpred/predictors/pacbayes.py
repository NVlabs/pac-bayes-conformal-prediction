from typing import Optional

import numpy as np
import torch
import functorch as ft
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from ..base import ConfPredictor, PredictionSet, ScoreFunction
from ..paramdists import GaussianParamDist
from ..softsort.neural_sort import soft_quantile
from ..utils import BufferList


class PACBayesConfPred(ConfPredictor):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        score_function: ScoreFunction,
        alpha: float,
        delta: float,
        alpha_hat: float,
        num_samples: int = 10,
        optimize_model: bool = False,
        optimize_score_function: bool = True,
        model_std_scale: float = 0.01,
        score_std_scale: float = 0.01,
        posterior_scale_factor: float = 1.0,
        optimize_bound: bool = False,
        prior_opt: str = "erm",
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

        self.has_constraint = True
        self.has_loss = True
        self.has_prior_constraint = False
        self.has_prior_loss = True

        self.backbone = backbone

        # functionalize both the model and the score function
        model_fn, model_params = ft.make_functional(head)
        self.model_fn = ft.vmap(model_fn)  # add batch dim to account for sampled models
        score_function_fn, score_function_params = ft.make_functional(score_function)
        self.score_function_fn = ft.vmap(score_function_fn)

        self.nonconformity_fn = score_function.nonconformity
        self.level_set_fn = score_function.level_set

        # initialize parameters for diagonal gaussian prior
        self.model_prior = GaussianParamDist(
            model_params, std_scale=model_std_scale
        )
        self.score_function_prior = GaussianParamDist(
            score_function_params, std_scale=score_std_scale
        )
        
        self.posterior_scale_factor = posterior_scale_factor
        
        self.optimize_bound = optimize_bound
        self.prior_opt = prior_opt

        # likewise for posterior -- this is what we need to differentiate through
        self.model_posterior = GaussianParamDist(
            model_params, std_scale=model_std_scale*posterior_scale_factor
        )
        self.score_function_posterior = GaussianParamDist(
            score_function_params, std_scale=score_std_scale*posterior_scale_factor
        )

        # For each sample of the parameters, we will have a different score quantile value
        self._num_samples = num_samples
        self._model_posterior_samples = BufferList(
            self.model_posterior.sample((num_samples,))
        )
        self._score_function_posterior_samples = BufferList(
            self.score_function_posterior.sample((num_samples,))
        )
        self.register_buffer(
            "threshold", torch.full((num_samples,), torch.nan, dtype=torch.float)
        )

    def _model_params(self):
        return self.model_posterior.parameters()

    def _score_params(self):
        return self.score_function_posterior.parameters()
    
    def _model_prior_params(self):
        if self.prior_opt == "erm":
            return self.model_prior.mu.parameters()
        elif self.prior_opt == "bbb":
            return self.model_posterior.parameters()
        else:
            raise ValueError

    def _score_prior_params(self):
        if self.prior_opt == "erm":
            return self.score_function_prior.mu.parameters()
        elif self.prior_opt == "bbb":
            return self.score_function_posterior.parameters()
        else:
            raise ValueError

    def _kl_div(self) -> Tensor:
        total_kl_div = self.model_posterior.kl_divergence(self.model_prior)
        total_kl_div += self.score_function_posterior.kl_divergence(
            self.score_function_prior
        )
        return total_kl_div
    
    def _bound_on_Ez(self,n:int):
        k = np.floor(self.alpha_hat*(n+1))
        exp_term = np.exp(1./(12*n-12) - 1./(12*k-11) - 1./(12*n-12*k+1))
        return n*np.sqrt((n-1)/(2*np.pi)/(k-1)/(n-k))*exp_term

    def _kl_bound(self, n: int) -> Tensor:
        k = np.floor(self.alpha_hat*(n+1))
        a = (k-1)/(n-1)
        binary_kl = a*np.log(a/self.alpha) + (1-a)*np.log((1-a)/(1-self.alpha))
        return (n-1)*binary_kl-np.log(self._bound_on_Ez(n)/self.delta)

    def _alpha_star(self, n: int) -> float:
        a = self.alpha - np.sqrt(
            (self._kl_div().item() + np.log(self._bound_on_Ez(n) / self.delta)) / 2 / (n - 1)
        )
        return (a * (n - 1) - 1) / (n + 1)

    def score_inputs(self, inputs: Tensor, sample=False, prior_mean=False):
        if sample:
            # sample parameters
            model_param_samples = self.model_posterior.sample((self._num_samples,))
            score_param_samples = self.score_function_posterior.sample(
                (self._num_samples,)
            )
        elif prior_mean:
            model_param_samples = [p.unsqueeze(0) for p in self.model_prior.mu]
            score_param_samples = [p.unsqueeze(0) for p in self.score_function_prior.mu ]
        else:
            model_param_samples = [p.data for p in self._model_posterior_samples]
            score_param_samples = [
                p.data for p in self._score_function_posterior_samples
            ]
        
        n_samples = model_param_samples[0].shape[0]
        # add dim for weight samples, [B, n, d_i]
        
        features = self.backbone(inputs)
        inputs = inputs.expand(n_samples, *inputs.shape)
        features = features.expand(n_samples, *features.shape)

        # compute score functions for all samples and all param samples
        outputs = self.model_fn(model_param_samples, features)  # [B, n, d_o]
        score_inputs = self.score_function_fn(
            score_param_samples, inputs, features, outputs
        )  # [B, n, d_s]
        return score_inputs

    def scores(self, inputs: Tensor, targets: Tensor, sample=False, prior_mean=False):
        score_inputs = self.score_inputs(inputs, sample=sample, prior_mean=prior_mean)
        targets = targets.expand(score_inputs.shape[0], *targets.shape)
        return self.nonconformity_fn(score_inputs, targets)  # [B, n]
    
    def prior_loss(self, inputs: Tensor, targets: Tensor, N: int) -> Tensor:
        """
        Trains prior mean greedily
        """
        n = inputs.shape[0]

        if self.prior_opt == "erm":
            
            score_inputs = self.score_inputs(inputs, prior_mean=True)
            targets = targets.expand(score_inputs.shape[0], *targets.shape)
            scores = self.nonconformity_fn(score_inputs, targets)  # [1, n]

            q = self._q(n)  # empirical coverage
            threshold = torch.quantile(scores, q, dim=1).unsqueeze(-1)
            pred_set = self.level_set_fn(score_inputs, threshold)

            # loss is mean size of prediction sets
            return pred_set.differentiable_volume().mean()
        elif self.prior_opt == "bbb":
            score_inputs = self.score_inputs(inputs, sample=True)
            targets = targets.expand(score_inputs.shape[0], *targets.shape)
            scores = self.nonconformity_fn(score_inputs, targets)  # [B, n]

            q = self._q(n)  # empirical coverage
            threshold = soft_quantile(scores, q, dim=1).unsqueeze(-1)
            pred_set = self.level_set_fn(score_inputs, threshold)

            # loss is mean size of prediction sets
            loss = pred_set.differentiable_volume().mean()
            reg = 1./N * self._kl_div()
            
            return loss #+ 1e-2*reg
        else:
            raise ValueError
            
    
    def post_prior_opt(self):
        """
        Initialize posterior around learned prior
        """
        if self.prior_opt == "erm":
            # we optimized the prior mean, so copy that over
            for post_mu, prior_mu in zip(self.model_posterior.mu.parameters(), self.model_prior.mu.parameters()):
                post_mu.data = prior_mu.data.clone()
                
            for post_mu, prior_mu in zip(self.score_function_posterior.mu.parameters(), self.score_function_prior.mu.parameters()):
                post_mu.data = prior_mu.data.clone()
        elif self.prior_opt == "bbb":
            # we optimized the posterior, so set that as the new prior
            for post_param, prior_param in zip(self.model_posterior.parameters(), self.model_prior.parameters()):
                prior_param.data = post_param.data.clone()
                
            for post_param, prior_param in zip(self.score_function_posterior.parameters(), self.score_function_prior.parameters()):
                prior_param.data = post_param.data.clone()

    def loss_and_constraint(self, inputs: Tensor, targets: Tensor, N: int) -> Tensor:
        """
        Implements batch training objective, here to minimize the expected set size

        Args:
            inputs (Tensor): batch of inputs to the model, [B, d_x]
            targets (Tensor): batch of targets [B, d_y]
            N (int): total size of dataset

        Returns:
            Tensor: Loss for batch (scalar)
        """
        n = inputs.shape[0]

        score_inputs = self.score_inputs(inputs, sample=True)
        targets = targets.expand(score_inputs.shape[0], *targets.shape)
        scores = self.nonconformity_fn(score_inputs, targets)  # [B, n]

        q = self._q(n)  # empirical coverage
        threshold = soft_quantile(scores, q, dim=1).unsqueeze(-1)
        pred_set = self.level_set_fn(score_inputs, threshold)

        # loss is mean size of prediction sets
        loss = pred_set.differentiable_volume().mean()
        
        # if we're optimzing the PAC bound, add regularization term
        if self.optimize_bound:
            loss += 1./(2*N) * self._kl_div()
        
        constraint = self._kl_div() - self._kl_bound(N)

        return loss, constraint

    def _update_posterior_samples(self):
        self._model_posterior_samples.replace(
            self.model_posterior.sample((self._num_samples,))
        )
        self._score_function_posterior_samples.replace(
            self.score_function_posterior.sample((self._num_samples,))
        )
        
    def calibrate(self, dataset: Dataset, **dl_kwargs: Optional[dict]) -> None:
        self._update_posterior_samples()
        
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
        
        scores = torch.concat(all_scores, dim=1)

        q = self._q(N, self.alpha_hat)
        self.threshold.data = torch.quantile(scores, q, dim=1)

    def forward(
        self, inputs: Tensor, score_quantile: Optional[Tensor] = None
    ) -> PredictionSet:
        if score_quantile is None:
            if torch.isnan(self.threshold).any():
                raise ValueError("Model not calibrated! Must call self.calibrate first")

            score_quantile = self.threshold

        score_inputs = self.score_inputs(inputs, sample=False)
        pred_set = self.level_set_fn(score_inputs, score_quantile.unsqueeze(-1))
        return pred_set
