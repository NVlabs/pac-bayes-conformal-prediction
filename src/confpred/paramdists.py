import math
from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out

from .utils import gaussian_kl_div


class GaussianParamDist(nn.Module):
    """
    Defines a Gaussian distribution the parameters of a nn.Module
    """

    def __init__(
        self,
        parameters: nn.ParameterList,
        zero_mean: bool = False,
        std_scale: float = 1.0,
        requires_grad: Optional[bool] = None,
    ) -> None:
        super().__init__()
        
        self.mu = nn.ParameterList(
            self._init_mean(p, zero_mean, requires_grad) for p in parameters
        )
        self.lnvar = nn.ParameterList(
            self._init_lnvar(p, std_scale, requires_grad) for p in parameters
        )

    def _init_mean(
        self,
        p: nn.Parameter,
        zero_mean: bool = False,
        requires_grad: Optional[bool] = None,
    ) -> nn.Parameter:
        rg = p.requires_grad if requires_grad is None else requires_grad
        if zero_mean:
            param = nn.Parameter(torch.zeros_like(p.data), requires_grad=rg)
        else:
            param = nn.Parameter(p.data.clone(), requires_grad=rg)
        return param

    def _init_lnvar(
        self, p: nn.Parameter, std_scale: float, requires_grad: Optional[bool] = None
    ) -> nn.Parameter:
        rg = p.requires_grad if requires_grad is None else requires_grad
        # scale variance by inferring size:
        if p.ndim > 1:
            fan_in, _ = _calculate_fan_in_and_fan_out(p.data)
            std_scale /= math.sqrt(fan_in)
        else:
            std_scale /= p.shape[0]
            
        param = nn.Parameter(torch.zeros_like(p.data) + math.log(std_scale), requires_grad=rg)
        return param

    @property
    def parameter_dists(self):
        return zip(self.mu, self.lnvar)

    def sample(self, batch_shape: Tuple[int] = (1,)) -> List[torch.Tensor]:
        """
        Samples from the distribution

        Args:
            batch_shape (Tuple[int], optional): shape of samples to take. Defaults to (1,).

        Returns:
            torch.List[nn.Parameter]: sample parameters
        """
        sample_parameters = []
        for mu, lnvar in zip(self.mu, self.lnvar):
            shape = batch_shape + mu.shape
            sample_parameters.append(
                mu + torch.exp(lnvar * 0.5) * torch.randn(shape, device=mu.device)
            )
        return sample_parameters

    def kl_divergence(self, other: "GaussianParamDist") -> torch.Tensor:
        """
        Args:
            other (GaussianParamDist)

        Returns:
            torch.Tensor: D_KL(self || other) (scalar)
        """
        if len(self.mu) == 0:
            return 0.
        
        if len(other.mu) != len(self.mu):
            raise ValueError(
                "other must be a distribution over the same parameters as self"
            )

        parameter_kl_divs = []
        for ((mu1, lnvar1), (mu2, lnvar2)) in zip(
            self.parameter_dists, other.parameter_dists
        ):
            parameter_kl_divs.append(
                gaussian_kl_div(mu1, lnvar1, mu2, lnvar2, var_is_logvar=True)
            )

        return torch.sum(torch.stack(parameter_kl_divs))