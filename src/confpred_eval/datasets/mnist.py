import numpy as np
import torch
from torch.utils.data import Subset
import os
from torchvision.datasets import MNIST as TVMNIST
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, Normalize, ToTensor, GaussianBlur, RandomRotation
from torchvision.utils import make_grid

from confpred.base import ConfPredictor
from confpred.pred_sets import DiscreteSet
from matplotlib.axes import Axes

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

VAL_TEST_TRANSFORMS = [AddGaussianNoise(std=1.3), RandomRotation(30)]

def create_val_test_dataset(data_root="~/datasets/mnist_transformed/"):
    transforms = [ToTensor(), Normalize((0.1307,), (0.3081,))]
    transforms = transforms + VAL_TEST_TRANSFORMS
    
    mnist = TVMNIST(
        root="~/datasets/",
        train=False,
        transform=Compose(transforms),
        download=True,
    )
    total_len = len(mnist)
    
    data_root = os.path.expanduser(data_root)
    print(f"Saving data to {data_root}")
    for i, (img, label) in enumerate(mnist):
        filename = os.path.join(data_root,str(label),f"{i:04d}.pt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(img, filename)
        
    
        


class MNIST(Subset):
    def __init__(self, N=5000, seed=1, split="train") -> None:
        total_val = 5000
        
        if split == "train":
            transforms = [ToTensor(), Normalize((0.1307,), (0.3081,))]

            base_dataset = TVMNIST(
                root="~/datasets/",
                train=not (split == "test"),
                transform=Compose(transforms),
                download=True,
            )
        else:
            base_dataset = DatasetFolder(
                root="~/datasets/mnist_transformed",
                loader=torch.load,
                extensions=("pt",),
                target_transform=int
            )

        datalen = len(base_dataset)
        np_random = np.random.RandomState(seed)
        indices = np_random.permutation(datalen)
        if split=="val":
            indices = indices[:total_val]
        elif split =="test":
            indices = indices[total_val:]
        
        indices = indices[:N]
        
        super().__init__(base_dataset, indices)

        

    def plot_predictions(self, ax: Axes, confpred: ConfPredictor, indices=None, **kwargs):
        images = []

        offset = 0
        padding = 2
        
        if indices is None:
            indices = np.random.choice(len(self),4)
        
        for j in indices:
            image, target = self[j]
            images.append(image)

            predset: DiscreteSet = confpred.forward(image[None,...])
            
            in_set : np.ndarray = predset.membership_scores.detach().numpy() > 0.5
            # remove batch dim
            in_set = in_set[...,0,:]
            # randomly pick a sample if our output has a batch over samples
            if in_set.ndim == 2:
                in_set = in_set[np.random.randint(in_set.shape[0])]
            
            items = np.argwhere(in_set).flatten()
            
            
            ax.text(offset,0,f"pred={items}\ntarget={target}",verticalalignment="bottom")
            offset += image.shape[1] + 2*padding

        grid = make_grid(images, padding=padding, pad_value=1.).detach().numpy()

        ax.imshow(np.transpose(grid,[1,2,0]))


if __name__ == "__main__":
    create_val_test_dataset()
    