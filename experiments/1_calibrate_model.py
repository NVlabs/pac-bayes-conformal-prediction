import os
import omegaconf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np


from confpred.base import ConfPredictor
from confpred.optim import unconstrained_opt, aug_lag_constrained_opt, iterative_penalty_opt


def optimize_prior(
    confpred_model: ConfPredictor,
    dataset: Dataset,
    cfg: omegaconf.DictConfig,
):
    n_val = len(dataset)

    if cfg.optim.cuda:
        confpred_model = confpred_model.cuda()

    dataloader = DataLoader(dataset, **cfg.dl_kwargs)
    
    constraint_opt_func = aug_lag_constrained_opt
    if cfg.optim.alg == "iterative_penalty":
        constraint_opt_func = iterative_penalty_opt
    elif cfg.optim.alg == "aug_lag":
        constraint_opt_func = aug_lag_constrained_opt
    else:
        print(f"alg={cfg.optim.alg} not understood, using aug_lag")

    # check whether we optimize this with a constraint or not
    if confpred_model.has_prior_constraint:
        constraint_opt_func(
            lambda X, Y: confpred_model.prior_loss_and_constraint(X, Y, n_val),
            dataloader,
            confpred_model.prior_params_to_optimize(),
            lr=cfg.optim.lr,
            max_outer_iter=cfg.optim.max_outer_iter,
            max_iter=cfg.optim.max_iter,
            term_threshold=cfg.optim.term_threshold,
        )
    else:
        unconstrained_opt(
            lambda X, Y: confpred_model.prior_loss(X, Y, n_val),
            dataloader,
            confpred_model.prior_params_to_optimize(),
            lr=cfg.optim.lr,
            max_iter=cfg.optim.max_iter,
            term_threshold=cfg.optim.term_threshold,
        )

    if cfg.optim.cuda:
        confpred_model.cpu()

def optimize(
    confpred_model: ConfPredictor,
    dataset: Dataset,
    cfg: omegaconf.DictConfig,
):
    n_val = len(dataset)

    if cfg.optim.cuda:
        confpred_model = confpred_model.cuda()

    dataloader = DataLoader(dataset, **cfg.dl_kwargs)
    
    constraint_opt_func = aug_lag_constrained_opt
    if cfg.optim.alg == "iterative_penalty":
        constraint_opt_func = iterative_penalty_opt
    elif cfg.optim.alg == "aug_lag":
        constraint_opt_func = aug_lag_constrained_opt
    else:
        print(f"alg={cfg.optim.alg} not understood, using aug_lag")

    # check whether we optimize this with a constraint or not
    if confpred_model.has_constraint:
        constraint_opt_func(
            lambda X, Y: confpred_model.loss_and_constraint(X, Y, n_val),
            dataloader,
            confpred_model.params_to_optimize(),
            lr=cfg.optim.lr,
            max_outer_iter=cfg.optim.max_outer_iter,
            max_iter=cfg.optim.max_iter,
            term_threshold=cfg.optim.term_threshold,
            lr_decay=cfg.optim.lr_decay
        )
    else:
        unconstrained_opt(
            lambda X, Y: confpred_model.loss(X, Y, n_val),
            dataloader,
            confpred_model.params_to_optimize(),
            lr=cfg.optim.lr,
            max_iter=cfg.optim.max_iter,
            term_threshold=cfg.optim.term_threshold,
        )

    if cfg.optim.cuda:
        confpred_model.cpu()


@hydra.main(config_path="conf/", config_name="defaults")
def main(cfg: omegaconf.DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset: torch.utils.data.Dataset = instantiate(
        cfg.data.dataset, N=cfg.data.split.val.N, seed=cfg.data.split.val.seed, split="val"
    )
    model: torch.nn.Module = instantiate(cfg.model)

    load_path = os.path.join(cfg.chkpt_path, cfg.name, "base.pt")
    load_path = hydra.utils.to_absolute_path(load_path)
    print("Loading model from", load_path)
    model.load_state_dict(torch.load(load_path))

    confpred: ConfPredictor = instantiate(cfg.calibrate.wrapper, backbone=model.backbone, head=model.head)

    # Split data if needed
    if cfg.calibrate.prior_split > 0:
        n_prior = int(np.floor(cfg.calibrate.prior_split*len(dataset)))
        indices = np.arange(len(dataset))
        prior_dataset = Subset(dataset, indices[:n_prior])
        posterior_dataset = Subset(dataset, indices[n_prior:])
        print(f"Holding {len(prior_dataset)} datapoints for prior optimization.")
        print(f"Keeping {len(posterior_dataset)} datapoints for calibration.")
    else:
        prior_dataset = None
        posterior_dataset = dataset
        
    if confpred.alpha_hat == -1:
        confpred.alpha_hat = confpred._alpha_star(len(posterior_dataset))
        print(f"Unspecified alpha_hat, using {confpred.alpha_hat=}")
    elif confpred.alpha_hat == -2:
        # use binomial bound version
        confpred.alpha_hat = confpred._alpha_star(len(posterior_dataset), binom=True)
        print(f"Unspecified alpha_hat, using {confpred.alpha_hat=}")

    if prior_dataset is not None and cfg.calibrate.optimize and confpred.has_prior_loss:
        print(f"Optimizing prior using alpha_hat={confpred.alpha_hat}")
        optimize_prior(confpred, prior_dataset, cfg.calibrate)
        confpred.post_prior_opt()

    if cfg.calibrate.optimize and confpred.has_loss:
        print(f"Optimizing posterior using alpha_hat={cfg.calibrate.alpha_hat}")
        optimize(confpred, posterior_dataset, cfg.calibrate)

    print(
        f"Calibrating model to ensure miscoverage rate only exceeds alpha={cfg.calibrate.alpha}, w.p. less than delta={cfg.calibrate.delta}"
    )
    confpred.calibrate(posterior_dataset, **cfg.calibrate.dl_kwargs)

    save_dir = os.path.join(
        cfg.chkpt_path, cfg.name, HydraConfig.get().job.override_dirname
    )
    save_dir = hydra.utils.to_absolute_path(save_dir)
    model_path = os.path.join(save_dir, "model.pt")
    config_path = os.path.join(save_dir, "config.yaml")
    print(f"Saving calibrated model to {model_path} along with config.")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(confpred.state_dict(), model_path)
    with open(config_path, "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
