import torch 
from torch import Tensor
from torch import nn
import functorch as ft

from confpred.paramdists import GaussianParamDist


class BNN(nn.Module):
    def __init__(self, model: nn.Module, loglikelihood: nn.Module, N: int, model_std_scale: float = 1e-1, num_samples: int = 10) -> None:
        super().__init__()
        
        self.backbone = model.backbone
        self.head = model.head
        
        self.loglikelihood = loglikelihood
        self.N = N
        
        # functionalize both the model and the score function
        model_fn, model_params = ft.make_functional(self.head)
        self.model_fn = ft.vmap(model_fn)  # add batch dim to account for sampled models
        
        # initialize parameters for diagonal gaussian prior
        self.model_prior = GaussianParamDist(
            model_params, std_scale=model_std_scale
        )
        
        # likewise for posterior -- this is what we need to differentiate through
        self.model_posterior = GaussianParamDist(
            model_params, std_scale=model_std_scale
        )
        
        # For each sample of the parameters, we will have a different score quantile value
        self._num_samples = num_samples
        
    def forward(self, inputs: Tensor):
        model_param_samples = self.model_posterior.sample((self._num_samples,))
        n_samples = model_param_samples[0].shape[0]

        features = self.backbone(inputs)
        features = features.expand(n_samples, *features.shape)
        outputs = self.model_fn(model_param_samples, features)  # [B, n, d_o]
        
        return outputs
    
    def loss(self, outputs, targets):
        n_samples = outputs.shape[0]
        targets = targets.expand(n_samples, *targets.shape)
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.reshape(-1)
        exp_ll = self.loglikelihood(outputs, targets)
        exp_ll = exp_ll.mean()
        return exp_ll + 1./self.N * self.model_posterior.kl_divergence(self.model_prior)
    