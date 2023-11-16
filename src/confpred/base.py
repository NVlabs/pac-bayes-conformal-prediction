from abc import abstractmethod
from typing import List, Optional, Tuple
from warnings import warn

import numpy as np
from scipy.special import betainc, betaincinv
from torch import BoolTensor, Tensor, nn
from torch.utils.data import Dataset


class PredictionSet:
    """
    Base class for prediction sets

    a PredictionSet object can represent a batch of sets,
    outputs should have the same number of batch elements as the tensors
    used to construct the object
    """

    @abstractmethod
    def volume(self) -> Tensor:
        """
        Returns the true volume of the set
        """
        raise NotImplementedError

    def differentiable_volume(self) -> Tensor:
        """
        Returns a differentiable surrogate of the true volume
        If not overridden, assumes self.volume() is differentiable already
        """
        return self.volume()

    @abstractmethod
    def in_set(self, element: Tensor) -> BoolTensor:
        """
        Checks if element is in the set described by set_params.
        """
        raise NotImplementedError


class ScoreFunction(nn.Module):
    """
    Base class for a NonconformityScoreFunction
    """

    @abstractmethod
    def forward(self, inputs: Tensor, features: Tensor, outputs: Tensor) -> Tensor:
        """
        Computes input / model output dependent portion of the nonconformity
        score function. Keeping this separate so that it can be easily
        vmapped for stochastic computation.
        """
        raise NotImplementedError

    @abstractmethod
    def nonconformity(self, score_inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Computes nonconformity scores s(x,f,y)
        for datapoints (inputs, targets),
        optionally also using a base model's predictions: outputs = f(inputs)

        Assumes all arguments contain a batch dimension

        Args:
            score_inputs (Tensor): input dependent, trainable component of nonconformity score
                output of self.forward()
                shape [..., d_s]
            targets (Tensor): targets from dataset
                shape [..., d_t]
        Returns:
            Tensor: [...] nonconformity score s(score_inputs, targets)
        """
        raise NotImplementedError

    @abstractmethod
    def level_set(self, score_inputs: Tensor, threshold: Tensor) -> PredictionSet:
        """
        Computes the sublevel set over the target space at given threshold t:

        C(x,t) = { y \in \mathcal{Y} | s(x,y) < t }

        Args:
            score_inputs (Tensor): input dependent, trainable component of nonconformity score
                output of self.forward()
                shape [..., d_s]
            threshold (Tensor):
                shape [...] (must broadcast with batch of score_inputs)

        Returns:
            PredictionSet, with batch dim [...]
        """
        raise NotImplementedError


class ConfPredictor(nn.Module):
    """
    Base class for conformal prediction wrapper
    """

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
        **kwargs,
    ):
        """
        Initializes a conformal prediction wrapper around a base model.
        Calibrates in order to ensure that the test-time miscoverage rate
        is less than alpha, with probability greater than 1-delta.

        Optionally, allows optimizing the resulting prediction sets.
        During optimization, we calibrate to achieve alpha_hat miscoverage
        on the data used for optimization, with alpha_hat typically less
        than alpha to ensure generalization to alpha miscoverage on test data.

        Args:
            backbone (nn.Module): non-randomized backbone of base predictor
            head (nn.Module): head of base predictor
            score_function (ScoreFunction): Nonconformity score function
            alpha (float): Desired miscoverage rate.
            delta (float): Desired bound on prob of guarantee failure
            alpha_hat (float): Desired empirical coverage to attain during training
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.score_function = score_function
        self.alpha = alpha
        self.delta = delta
        self.alpha_hat = alpha_hat

        self.optimize_model = optimize_model
        self.optimize_score_function = optimize_score_function

        self.has_prior_loss = False
        self.has_prior_constraint = False
        self.has_loss = False
        self.has_constraint = False

    def _model_params(self):
        return self.head.parameters()

    def _score_params(self):
        return self.score_function.parameters()

    def _model_prior_params(self):
        return self.head.parameters()

    def _score_prior_params(self):
        return self.score_function.parameters()

    def prior_params_to_optimize(self) -> List[nn.Parameter]:
        """
        Returns nn.Parameters to optimize
        """
        params = []
        if self.optimize_model:
            params.extend(self._model_prior_params())
        if self.optimize_score_function:
            params.extend(self._score_prior_params())

        return params

    def params_to_optimize(self) -> List[nn.Parameter]:
        """
        Returns nn.Parameters to optimize
        """
        params = []
        if self.optimize_model:
            params.extend(self._model_params())
        if self.optimize_score_function:
            params.extend(self._score_params())

        return params

    def _q(self, n: int, alpha_hat: Optional[float] = None) -> float:
        """
        Returns quantile of a dataset of length n
        closest to but above the alpha_hat quantile while

        Args:
            n (int): length of dataset
            alpha_hat (Optional[float], optional): Quantile desired, if None, uses self.alpha_hat. Defaults to None.

        Returns:
            float: desired quantile
        """
        if alpha_hat is None:
            alpha_hat = self.alpha_hat
        q = np.ceil((n + 1) * (1 - alpha_hat)) / n
        if q > 1:
            warn(f"Warning, trying to access {q} quantile, reverting to q=1")
            return 1.0
        else:
            return q

    def _alpha_star(self, n: int, binom=False) -> float:
        """
        Returns required empirical coverage on a dataset of n examples
        in order to ensure test-time miscoverage remains below self.alpha
        with probability greater than 1-self.delta

        As implemented, returns alpha_star assuming
        calibration data hasn't been also used to optimize any other
        parameters.

        Args:
            n (int): size of calibration set
        """
        if binom:
            alpha_stars = np.linspace(0, self.alpha, 100)
            # ks = np.floor(alpha_stars*(n+1) - 1)
            # cdf = betainc(n - ks, 1 + ks, 1 - self.alpha)
            # idx = (cdf > self.delta)*(-1) + (cdf <= self.delta)*np.arange(100)
            # return alpha_stars[np.max(idx)]
            v = np.floor((n + 1) * alpha_stars)
            guarantee_prob = betaincinv(n + 1 - v, v, self.delta)
            alpha_stars[guarantee_prob < 1 - self.alpha] = 0
            return alpha_stars.max()
        else:
            return self.alpha - np.sqrt(-np.log(self.delta) / (2 * n))

    def score_inputs(self, inputs: Tensor) -> Tensor:
        """
        Returns input-dependent terms of score function,
        everything that is needed to compute level sets

        Args:
            inputs (Tensor): inputs (..., d_x)

        Returns:
            Tensor: score_inputs (..., d_s)
        """
        features = self.backbone(inputs)
        outputs = self.head(features)
        score_inputs = self.score_function(inputs, features, outputs)
        return score_inputs

    def scores(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Returns nonconformity scores for given input/target pairs

        Args:
            inputs (Tensor): inputs (..., d_x)
            targets (Tensor): targets (..., d_y)

        Returns:
            Tensor: score values (...)
        """
        score_inputs = self.score_inputs(inputs)  # (..., d_x)
        return self.score_function.nonconformity(score_inputs, targets)  # (...)

    @abstractmethod
    def loss(self, inputs: Tensor, targets: Tensor, N: int) -> Tensor:
        """
        Returns loss on prediction sets for inputs and targets
        Must implement if self.has_loss = True, but self.has_constraint = False
        Model will be calibrated using same data

        Args:
            inputs (Tensor): inputs (..., d_x)
            targets (Tensor): targets (..., d_y)
            N (int): total size of calibration dataset

        Returns:
            Tensor: loss (scalar)
        """
        raise NotImplementedError

    @abstractmethod
    def loss_and_constraint(
        self, inputs: Tensor, targets: Tensor, N: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns loss on prediction sets for inputs and targets
        As well as constraint value

        Optimizer will minimize loss subj to constraint <= 0

        Must implement if self.has_loss = True, and self.has_constraint = True
        Model will be calibrated using same data

        Args:
            inputs (Tensor): inputs (..., d_x)
            targets (Tensor): targets (..., d_y)
            N (int): total size of calibration dataset

        Returns:
            Tuple[Tensor, Tensor]: ( loss (scalar), constraint (scalar) )
        """
        raise NotImplementedError

    @abstractmethod
    def prior_loss(self, inputs: Tensor, targets: Tensor, N: int) -> Tensor:
        """
        Returns loss on prediction sets for inputs and targets
        Model will be calibrated on fresh data not fed to this loss.
        Must implement if self.has_prior_loss = True, but self.has_prior_constraint = False

        Args:
            inputs (Tensor): inputs (..., d_x)
            targets (Tensor): targets (..., d_y)
            N (int): total size of calibration dataset

        Returns:
            Tensor: loss (scalar)
        """
        raise NotImplementedError

    @abstractmethod
    def prior_loss_and_constraint(
        self, inputs: Tensor, targets: Tensor, N: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns loss on prediction sets for inputs and targets
        As well as constraint value

        Optimizer will minimize loss subj to constraint <= 0

        Model will be calibrated on fresh data not fed to this loss.
        Must implement if self.has_prior_loss = True, and self.has_prior_constraint = True

        Args:
            inputs (Tensor): inputs (..., d_x)
            targets (Tensor): targets (..., d_y)
            N (int): total size of calibration dataset

        Returns:
            Tuple[Tensor, Tensor]: ( loss (scalar), constraint (scalar) )
        """
        raise NotImplementedError

    def post_prior_opt(self):
        """
        Called after optimizing prior, useful for initializing variables
        for posterior optimization.
        """
        pass

    @abstractmethod
    def calibrate(self, dataset: Dataset, **dl_kwargs) -> None:
        """
        Updates self's parameters to calibrate prediction sets for
        calibration data given by inputs/targets

        Args:
            dataset (Dataset): torch dataset to calibrate on
            **dl_kwargs: keyword arguments passed to dataloader
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor) -> PredictionSet:
        """
        Returns prediction sets for a batch of inputs

        Args:
            inputs (Tensor): inptus (..., d_x)

        Returns:
            PredictionSet: with batch shape (...)
        """
        raise NotImplementedError
