import os
from confpred.base import ConfPredictor, PredictionSet
from confpred.predictors import PACBayesConfPred
import omegaconf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np


def eval_model(
    confpred_model: ConfPredictor,
    dataset: Dataset,
    eval_cfg: omegaconf.DictConfig,
):
    cuda = eval_cfg.cuda

    if cuda:
        confpred_model = confpred_model.cuda()

    dataloader = DataLoader(dataset, **eval_cfg.calibrate.dl_kwargs)
    all_scores = []
    all_coverages = []
    all_set_sizes = []
    for inputs, targets in dataloader:
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        with torch.no_grad():
            test_preds: PredictionSet = confpred_model(inputs)
            all_scores.append(
                confpred_model.scores(inputs, targets).detach().cpu().numpy()
            )
            all_coverages.append(
                test_preds.in_set(targets).float().mean().detach().cpu().numpy()
            )
            all_set_sizes.append(test_preds.volume().mean().detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=-1)
    coverage = np.mean(all_coverages)
    set_size = np.mean(all_set_sizes)

    print(
        f" Desired test coverage: {1 - confpred_model.alpha}.\n Empirical coverage: {coverage}."
    )
    print(f" Average set volume on data: {set_size}")

    if cuda:
        confpred_model.cpu()

    return {
        "threshold": confpred_model.threshold.detach().numpy(),
        "scores": scores,
        "coverage": coverage,
        "set_size": set_size,
    }


@hydra.main(version_base=None, config_path="conf/", config_name="defaults")
def main(cfg: omegaconf.DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    val_dataset: torch.utils.data.Dataset = instantiate(
        cfg.data.dataset,
        N=cfg.data.split.val.N,
        seed=cfg.data.split.val.seed,
        split="val",
    )
    # Split data if needed
    if cfg.calibrate.prior_split > 0:
        n_prior = int(np.floor(cfg.calibrate.prior_split * len(val_dataset)))
        indices = np.arange(len(val_dataset))
        prior_dataset = Subset(val_dataset, indices[:n_prior])
        posterior_dataset = Subset(val_dataset, indices[n_prior:])
        print(f"Holding {len(prior_dataset)} datapoints for prior optimization.")
        print(f"Keeping {len(posterior_dataset)} datapoints for calibration.")
    else:
        prior_dataset = None
        posterior_dataset = val_dataset

    test_dataset: torch.utils.data.Dataset = instantiate(
        cfg.data.dataset,
        N=cfg.data.split.test.N,
        seed=cfg.data.split.test.seed,
        split="test",
    )
    model: torch.nn.Module = instantiate(cfg.model)
    load_path = os.path.join(cfg.chkpt_path, cfg.name, "base.pt")
    load_path = hydra.utils.to_absolute_path(load_path)
    print("Loading model from", load_path)
    model.load_state_dict(torch.load(load_path))

    confpred: ConfPredictor = instantiate(
        cfg.calibrate.wrapper, backbone=model.backbone, head=model.head
    )

    load_dir = os.path.join(
        cfg.chkpt_path, cfg.name, HydraConfig.get().job.override_dirname
    )
    model_path = os.path.join(load_dir, "model.pt")
    model_path = hydra.utils.to_absolute_path(model_path)
    print("Loading confpred model from", model_path)
    confpred.load_state_dict(torch.load(model_path))

    results = {}

    if prior_dataset is not None:
        print(f"Evaluating model on prior set.")
        results["prior"] = eval_model(confpred, prior_dataset, cfg)
    print(f"Evaluating model on posterior set.")
    results["posterior"] = eval_model(confpred, posterior_dataset, cfg)

    # check if constraint is satisfied
    n = len(posterior_dataset)
    if isinstance(confpred, PACBayesConfPred):
        constraint = confpred._kl_div() - confpred._kl_bound(n)
        results["constraint_valid"] = constraint < 0
        results["efficiency_bound"] = results["posterior"]["set_size"] + 10*(
            confpred._kl_div().detach().cpu().numpy()
            + np.log(2 * np.sqrt(n) / confpred.delta)
        ) / (2 * n)
    else:
        results["constraint_valid"] = True
        # use hoeffding
        results["efficiency_bound"] = results["posterior"]["set_size"] + 10*np.sqrt(1/(2*n)*np.log(2/confpred.delta))

    print(f"Evaluating model on test set.")
    results["test"] = eval_model(confpred, test_dataset, cfg)

    save_dir = os.path.join(
        cfg.results_path, cfg.name, HydraConfig.get().job.override_dirname
    )
    save_dir = hydra.utils.to_absolute_path(save_dir)
    results_path = os.path.join(save_dir, "results.pkl")
    config_path = os.path.join(save_dir, "config.yaml")
    print(f"Saving calibrated model to {model_path} along with config.")

    print(f"Saving results to {save_dir}")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save(results, results_path)
    with open(config_path, "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
