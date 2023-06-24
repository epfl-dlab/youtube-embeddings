import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

def plot_df_similarity(df, embed_data):
    df_tmp = df[df["embed_data"] == embed_data].copy()

    agg_dfs = []

    for agg_num in np.arange(4) + 2:
        df_tmp["agglist"] = df_tmp.apply(
            lambda row: [
                mode == exp
                for agree, exp, mode in zip(
                    row["agree_count"], row["exp_result"], row["mode_res"]
                )
                if agree >= agg_num
            ],
            axis=1,
        )
        agg_df = (
            df_tmp.groupby("embed")["agglist"]
            .apply(lambda x: np.mean(list(chain.from_iterable(x))))
            .to_frame("agg")
            .assign(agg_num=agg_num)
        )
        agg_dfs.append(agg_df)

    return pd.concat(agg_dfs).reset_index()


def plot_embed_similarity(ax, plot_df):
    sns.barplot(
        plot_df,
        x="agg_num",
        y="agg",
        hue="embed",
        ax=ax,
        hue_order=["reddit", "content", "recomm"],
    )

    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f")

    ax.get_legend().remove()


def fraction_plot_df(full_df_agreement, ax, fontsize=12):
    df_tmp = full_df_agreement

    df_tmp["agglist"] = df_tmp.apply(
        lambda row: [
            mode == exp
            for agree, exp, mode in zip(
                row["agree_count"], row["exp_result"], row["mode_res"]
            )
            if agree >= 2
        ],
        axis=1,
    )
    agg_df = (
        df_tmp.groupby(["embed", "frac"])["agglist"]
        .apply(lambda x: np.mean(list(chain.from_iterable(x))))
        .to_frame("agg")
        .reset_index()
    )

    if ax is not None:
        sns.barplot(
            agg_df,
            x="frac",
            hue="embed",
            y="agg",
            hue_order=["reddit", "content", "recomm"],
            ax=ax,
        )

        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.2f")

        ax.set_xlabel("q", fontsize=fontsize)
        ax.set_ylabel("Agreement between Workers and Embedding", fontsize=fontsize)
        ax.set_title(
            "Agreement between Workers and Embedding as a function of q", fontsize=fontsize
        )
        
    return agg_df