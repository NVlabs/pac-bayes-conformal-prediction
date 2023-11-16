import os
from typing import List

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import Subset


def load_confpred(chkpt_folder):
    model_path = os.path.join(chkpt_folder, "model.pt")
    config_path = os.path.join(chkpt_folder, "config.yaml")

    cfg = OmegaConf.load(config_path)
    model: torch.nn.Module = instantiate(cfg.model)
    load_path = os.path.join(cfg.chkpt_path, cfg.name, "base.pt")
    load_path = to_absolute_path(load_path)
    model.load_state_dict(torch.load(load_path))
    confpred = instantiate(
        cfg.calibrate.wrapper, backbone=model.backbone, head=model.head
    )
    confpred.load_state_dict(torch.load(model_path))

    return confpred


def load_dataset(chkpt_folder, split="test", post=True):
    config_path = os.path.join(chkpt_folder, "config.yaml")

    cfg = OmegaConf.load(config_path)
    dataset = instantiate(
        cfg.data.dataset,
        N=cfg.data.split[split].N,
        seed=cfg.data.split[split].seed,
        split=split,
    )
    if split == "val":
        # Split data if needed
        if cfg.calibrate.prior_split > 0:
            n_prior = int(np.floor(cfg.calibrate.prior_split * len(dataset)))
            indices = np.arange(len(dataset))
            prior_dataset = Subset(dataset, indices[:n_prior])
            posterior_dataset = Subset(dataset, indices[n_prior:])
            print(f"Holding {len(prior_dataset)} datapoints for prior optimization.")
            print(f"Keeping {len(posterior_dataset)} datapoints for calibration.")
        else:
            prior_dataset = None
            posterior_dataset = dataset

        if post:
            return posterior_dataset
        else:
            return prior_dataset
    return dataset


def load_results(chkpt_folder, df: pd.DataFrame = None):
    # Add results and config params as a row in the dataframe
    results_path = os.path.join(chkpt_folder, "results.pkl")
    config_path = os.path.join(chkpt_folder, "config.yaml")
    cfg = OmegaConf.load(config_path)
    results = torch.load(results_path)

    # flatten dict
    results = pd.json_normalize(results)

    # flatten config
    cfg_flattened = pd.json_normalize(OmegaConf.to_container(cfg, resolve=True))

    # merge into one row
    new_row = results.join(cfg_flattened)
    new_row["filename"] = chkpt_folder

    if df is not None:
        df = pd.concat([df, new_row], ignore_index=True)
        return df
    else:
        return new_row


def summarize_folders(folder_names, existing_df=None):
    df = existing_df
    for folder in folder_names:
        df = load_results(folder, df)
    return df
