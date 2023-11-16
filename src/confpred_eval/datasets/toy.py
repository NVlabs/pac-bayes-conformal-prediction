import numpy as np
import torch
from torch.utils.data import TensorDataset

from confpred.base import ConfPredictor
from matplotlib.axes import Axes

def noisy_fn(x, **kwargs):
    return torch.cos(5*x) + 0.3*(torch.rand(x.size(),**kwargs)-0.5) + 1.8*(torch.rand(x.size(),**kwargs)-0.5)*torch.sigmoid(5*x)

class ToyDataset(TensorDataset):
    def __init__(self, N=500, xmin=-1, xmax=1, seed: int = 1, split="train") -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.X = xmin + (xmax-xmin)*torch.rand(N,generator=rng)[:,None]
        self.Y = noisy_fn(self.X,generator=rng)
        super().__init__(self.X,self.Y)
        

    def plot_predictions(self, ax: Axes, confpred: ConfPredictor, color="C1", **kwargs) -> None:
        X = torch.linspace(-1,1,100)
        predsets = confpred.forward(X[:,None])
        mean_size = predsets.differentiable_volume().mean().item()
        if predsets.centroid.ndim == 3:
            pred_to_use = np.random.randint(predsets.centroid.shape[0], size=len(X))
            centroids = predsets.centroid.detach().numpy()[pred_to_use, np.arange(len(X))]
            radii = predsets.radius.detach().numpy()[pred_to_use, np.arange(len(X))]
            h, =ax.plot(X, centroids[:,0], color=color)
            ax.fill_between(
                X,
                centroids[:, 0] + radii,
                centroids[:, 0] - radii,
                alpha=0.3,
                color=h.get_color()
            )
        else:
            h, = ax.plot(X, predsets.centroid.detach()[:,0], color=color, **kwargs)
            ax.fill_between(
                X,
                predsets.centroid.detach()[:, 0] + predsets.radius.detach(),
                predsets.centroid.detach()[:, 0] - predsets.radius.detach(),
                alpha=0.2,
                color=h.get_color()
            )

        ax.set_ylim([-2.5,2.5])
        ax.set_xlim([-1,1])
        ax.scatter(self.X, self.Y, alpha=min(1., 50./len(self.X)),   marker="x", color='k', label="Cal. data")
