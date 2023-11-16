import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import omegaconf
import hydra
from hydra.utils import instantiate

from confpred.optim import unconstrained_opt


def train(
    model: nn.Module,
    loss: nn.Module,
    dataset: Dataset,
    train_cfg: omegaconf.DictConfig,
):
    cuda = train_cfg.cuda
    lr = train_cfg.lr
    max_iter = train_cfg.epochs
    term_threshold = train_cfg.term_threshold

    if cuda:
        model = model.cuda()

    dataloader = DataLoader(dataset, **train_cfg.dl_kwargs)

    if hasattr(model, "loss"):
        loss = lambda outputs,labels: model.loss(outputs,labels)
    
    def loss_func(X,Y):
        outputs = model(X)
        return loss(outputs, Y)
    
    unconstrained_opt(
        loss_func,
        dataloader,
        list(model.parameters()),
        lr=lr,
        max_iter=max_iter,
        term_threshold=term_threshold,
        # log_fn=wandb.log,
    )


@hydra.main(config_path="conf/", config_name="defaults")
def main(cfg: omegaconf.DictConfig) -> None:
    dataset = instantiate(cfg.data.dataset, N=cfg.data.split.train.N, seed=cfg.data.split.train.seed)
    loss = instantiate(cfg.data.loss)
    model = instantiate(cfg.model)

    train(model, loss, dataset, cfg.train)

    save_path = os.path.join(cfg.chkpt_path, cfg.name, "base.pt")
    save_path = hydra.utils.to_absolute_path(save_path)
    print("Saving model to:", save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
