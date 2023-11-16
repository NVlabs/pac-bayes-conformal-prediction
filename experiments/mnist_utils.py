from dataclasses import dataclass
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


# Utilities for filtering results dataframe for valid entries
@dataclass
class Filter:
    key: str  # column of df
    condition: Callable[[pd.Series], pd.Series]  # condtion on column's values


def apply_filters(df: pd.DataFrame, filters: List[Filter]) -> pd.DataFrame:
    for filter in filters:
        df = df[filter.condition(df[filter.key])]
    return df


common_filters = [
    Filter("constraint_valid", lambda x: x == True),
    Filter("seed", lambda x: np.logical_not(np.isnan(x))),
]

relevant_columns = [
    "calibrate.name",
    "calibrate.alpha",
    "calibrate.delta",
    "calibrate.prior_split",
    "data.split.val.N",
    "calibrate.alpha_hat",
    "posterior.set_size",
    "test.set_size",
    "posterior.coverage",
    "test.coverage",
    "calibrate.wrapper.prior_opt",
    "seed",
]


def process_confpred(df: pd.DataFrame) -> pd.DataFrame:
    filters = common_filters + [
        Filter("calibrate.name", lambda x: x == "confpred"),
        Filter("calibrate.alpha_hat", lambda x: x < 0),
        Filter("seed", lambda x: x.isin([2023, 2024, 2025, 2026, 2027])),
    ]
    df = apply_filters(df, filters)
    return df[relevant_columns]


def process_learnable(df: pd.DataFrame) -> pd.DataFrame:
    filters = common_filters + [
        Filter("calibrate.name", lambda x: x == "learnable"),
        Filter("calibrate.alpha_hat", lambda x: x < 0),
        Filter("seed", lambda x: x.isin([2023, 2024, 2025, 2026, 2027])),
    ]
    df = apply_filters(df, filters)
    return df[relevant_columns]


def process_pacbayes(df: pd.DataFrame, only_bbb=True) -> pd.DataFrame:
    filters = common_filters + [
        Filter("calibrate.name", lambda x: x == "pacbayes"),
        Filter("calibrate.delta", lambda x: x == 0.01),
    ]
    if only_bbb:
        filters += [Filter("calibrate.wrapper.prior_opt", lambda x: x == "bbb")]
    df = apply_filters(df, filters)
    # for each seed, N, and prior_split,
    # pick best performing alpha_hat based on efficiency bound
    df = (
        df.sort_values("efficiency_bound")
        .groupby(
            [
                "data.split.val.N",
                "calibrate.prior_split",
                "calibrate.wrapper.prior_opt",
                "seed",
            ]
        )
        .apply(pd.DataFrame.head, n=1)
        .reset_index(drop=True)
    )
    return df[relevant_columns]


def categorize_data(df: pd.DataFrame) -> pd.DataFrame:
    # account for different versions of alpha_hat selection
    df.loc[df["calibrate.alpha_hat"] == -2, "calibrate.name"] += "2b"
    # rename to nice names for plots
    df["calibrate.name"] = (
        df["calibrate.name"]
        .astype("category")
        .cat.rename_categories(
            {
                "confpred": "Standard (Vovk 2a)",
                "confpred2b": "Standard (Vovk 2b)",
                "learnable": "Learned (Vovk 2a)",
                "learnable2b": "Learned (Vovk 2b)",
                "pacbayes": "PAC-Bayes",
            }
        )
    )

    return df


def plot_coverage_vs_n(ax: Axes, results, prior_split=0.5, ylabel=False):
    # collate data
    total_df = pd.concat(
        [
            process_confpred(results),
            apply_filters(
                process_learnable(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
            apply_filters(
                process_pacbayes(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
        ]
    )
    total_df = categorize_data(total_df)

    # colors for plots
    colors = sns.color_palette("muted", 3)
    colors = [
        colors[0],
        sns.set_hls_values(colors[0], l=0.4),
        colors[1],
        sns.set_hls_values(colors[1], l=0.4),
        colors[2],
    ]

    sns.boxplot(
        total_df,
        x="data.split.val.N",
        y="test.coverage",
        hue="calibrate.name",
        ax=ax,
        palette=colors,
    )

    ax.legend()
    plt.tight_layout()
    ax.set_xlabel(r"Size of Calibration Dataset ($N$)")
    if ylabel:
        ax.set_ylabel("Average Prediction Set Size")
    else:
        ax.set_ylabel(None)


def plot_efficiency_vs_n(
    ax: Axes,
    results,
    prior_split=0.5,
    yval="test.set_size",
    errorbar=("ci", 95),
    ylabel=False,
):
    # collate data
    total_df = pd.concat(
        [
            process_confpred(results),
            apply_filters(
                process_learnable(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
            apply_filters(
                process_pacbayes(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
        ]
    )
    total_df = categorize_data(total_df)

    # colors for plots
    colors = sns.color_palette("muted", 3)
    colors = [
        colors[0],
        sns.set_hls_values(colors[0], l=0.4),
        colors[1],
        sns.set_hls_values(colors[1], l=0.4),
        colors[2],
    ]

    sns.barplot(
        total_df,
        x="data.split.val.N",
        y=yval,
        errorbar=errorbar,
        hue="calibrate.name",
        ax=ax,
        palette=colors,
        capsize=0.1,
        errwidth=1,
    )
    ax.legend()
    plt.tight_layout()
    ax.set_xlabel(r"Size of Calibration Dataset ($N$)")
    if ylabel:
        ax.set_ylabel("Average Prediction Set Size")
    else:
        ax.set_ylabel(None)


def plot_efficency_vs_coverage(ax: Axes, results, prior_split=0.5, ylabel=False):
    # collate data
    total_df = pd.concat(
        [
            process_confpred(results),
            apply_filters(
                process_learnable(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
            apply_filters(
                process_pacbayes(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
        ]
    )
    total_df = categorize_data(total_df)

    # colors for plots
    colors = sns.color_palette("muted", 3)
    colors = [
        colors[0],
        sns.set_hls_values(colors[0], l=0.4),
        colors[1],
        sns.set_hls_values(colors[1], l=0.4),
        colors[2],
    ]
    markers = ["o", "x", "o", "x", "o"]

    sns.scatterplot(
        total_df,
        x="test.coverage",
        y="test.set_size",
        hue="calibrate.name",
        palette=colors,
        markers=markers,
        ax=ax,
        # legend=None,
    )
    # ax.legend()
    plt.tight_layout()
    ax.set_xlabel("Observed Coverage Rate on Test Set")
    if ylabel:
        ax.set_ylabel("Average Prediction Set Size")
    else:
        ax.set_ylabel(None)


def plot_efficency_vs_alpha_hat(ax: Axes, results, prior_split=0.5, ylabel=False):
    # collate data
    total_df = pd.concat(
        [
            process_confpred(results),
            apply_filters(
                process_learnable(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
            apply_filters(
                process_pacbayes(results),
                [Filter("calibrate.prior_split", lambda x: x == prior_split)],
            ),
        ]
    )
    total_df["inferred_alpha_hat"] = 1 - total_df["posterior.coverage"]
    total_df = categorize_data(total_df)

    # colors for plots
    colors = sns.color_palette("muted", 3)
    colors = [
        colors[0],
        sns.set_hls_values(colors[0], l=0.4),
        colors[1],
        sns.set_hls_values(colors[1], l=0.4),
        colors[2],
    ]
    markers = ["o", "x", "o", "x", "o"]

    sns.scatterplot(
        total_df,
        x="inferred_alpha_hat",
        y="test.set_size",
        hue="calibrate.name",
        palette=colors,
        markers=markers,
        ax=ax,
    )
    ax.legend()
    plt.tight_layout()
    ax.set_xlabel(r"Empirical miscoverage rate $\hat{\alpha}$")
    if ylabel:
        ax.set_ylabel("Average Prediction Set Size")
    else:
        ax.set_ylabel(None)


def plot_relative_efficiency_vs_n(
    ax: Axes, results, prior_splits=[0.25, 0.5, 0.75], ylabel=False
):
    confpred_df: pd.DataFrame = process_confpred(results)
    naive_df = apply_filters(
        confpred_df, [Filter("calibrate.alpha_hat", lambda x: x == -1)]
    )
    naive_df = naive_df[["data.split.val.N", "test.set_size", "seed"]]
    naive_df = naive_df.rename(columns={"test.set_size": "baseline.test.set_size"})

    vovk2b_df = apply_filters(
        confpred_df, [Filter("calibrate.alpha_hat", lambda x: x == -2)]
    )
    vovk2b_df = vovk2b_df[["data.split.val.N", "test.set_size", "seed"]]

    total_df = pd.concat(
        [
            apply_filters(
                process_learnable(results),
                [Filter("calibrate.prior_split", lambda x: x.isin(prior_splits))],
            ),
            apply_filters(
                process_pacbayes(results),
                [Filter("calibrate.prior_split", lambda x: x.isin(prior_splits))],
            ),
        ]
    )
    total_df = categorize_data(total_df)

    # sort by prior split
    total_df = total_df.sort_values(by=["calibrate.prior_split", "calibrate.name"])
    # join standard approach data
    total_df = pd.merge(total_df, naive_df, on=["data.split.val.N", "seed"])
    total_df["improvement"] = (
        total_df["test.set_size"] / total_df["baseline.test.set_size"]
    )

    vovk2b_df = pd.merge(vovk2b_df, naive_df, on=["data.split.val.N", "seed"])
    vovk2b_df["improvement"] = (
        vovk2b_df["test.set_size"] / vovk2b_df["baseline.test.set_size"]
    )
    # colors for plots
    colors = sns.color_palette("muted", 3)
    colors = [
        colors[0],
        sns.set_hls_values(colors[0], l=0.4),
        colors[1],
        sns.set_hls_values(colors[1], l=0.4),
        colors[2],
    ]

    ax.axhline(1.0, color=colors[0], linestyle=":", label="Standard (Vovk 2a)")

    ax.axhline(
        vovk2b_df["improvement"].mean(),
        color=colors[1],
        linestyle="-.",
        label="Standard (Vovk 2b)",
    )

    sns.barplot(
        total_df,
        x="calibrate.prior_split",
        y="improvement",
        hue="calibrate.name",
        errorbar="se",
        ax=ax,
        palette=colors[2:],
        errwidth=0.5,
    )
    ax.legend(loc="upper right")
    ax.set_ylim([0.0, 1.7])
    ax.set_xlabel("Data Split")
    if ylabel:
        ax.set_ylabel("Relative Prediction Set Size")
    else:
        ax.set_ylabel(None)


def plot_efficiency_vs_prior_opts(ax: Axes, results, ylabel=False):
    colors = sns.color_palette("muted", 5)[-1:]
    colors += sns.color_palette("Paired", 6)

    selected_df = process_pacbayes(results, only_bbb=False)

    no_opt_df = selected_df[selected_df["calibrate.prior_split"] == 0.0]
    no_opt_df["calibrate.wrapper.prior_opt"] = "None"

    selected_df = selected_df[selected_df["calibrate.prior_split"] > 0.0]

    total_df = pd.concat([no_opt_df, selected_df])
    total_df["calibrate.wrapper.prior_opt"] = (
        total_df["calibrate.wrapper.prior_opt"]
        .astype("category")
        .cat.reorder_categories(["None", "erm", "bbb"], ordered=True)
    )
    total_df["calibrate.wrapper.prior_opt"] = total_df[
        "calibrate.wrapper.prior_opt"
    ].cat.rename_categories({"None": "None", "erm": r"$\mu$", "bbb": r"$\mu,\sigma^2$"})
    total_df["opt_split"] = total_df.apply(
        lambda x: x["calibrate.wrapper.prior_opt"]
        + f" {x['calibrate.prior_split']:0.2f}/{1 - x['calibrate.prior_split']:0.2f}",
        axis=1,
    )
    total_df = total_df.sort_values(
        ["calibrate.prior_split", "calibrate.wrapper.prior_opt"]
    )
    sns.barplot(
        total_df,
        x="data.split.val.N",
        y="test.set_size",
        hue="opt_split",
        palette=colors,
        ax=ax,
        capsize=0.1,
        errwidth=1,
    )
    ax.legend(loc="upper right")
    ax.set_ylim([1.0, 5.0])
    ax.set_ylabel("Average Prediction Set Size")
    ax.set_xlabel(r"Size of Calibration Dataset ($N$)")
