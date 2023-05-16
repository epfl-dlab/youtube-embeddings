import numpy as np
import pandas as pd
import umap
import plotly.express as px

from typing import Optional
from collections.abc import Iterable
from scipy.sparse import csr_matrix


def csr_from_df(
    matrix_df: pd.DataFrame,
    subreddit_col: str = "subreddit_id",
    channel_col: str = "channelId",
    count_col: str = "total",
    sub_c: pd.CategoricalDtype = None,
    cha_c: pd.CategoricalDtype = None,
):
    """Compute Co-occurrence csr matrix from dataframe

    Assumes a dataframe of three columns, subreddit_col, channel_col and count_col
    Where subreddit_col and channel_col form an index
    """

    if sub_c is None:
        sub_c = pd.CategoricalDtype(
            sorted(matrix_df[subreddit_col].unique()), ordered=True
        )

    if cha_c is None:
        cha_c = pd.CategoricalDtype(
            sorted(matrix_df[channel_col].unique()), ordered=True
        )

    row = matrix_df[subreddit_col].astype(sub_c).cat.codes
    col = matrix_df[channel_col].astype(cha_c).cat.codes
    sparse_matrix = csr_matrix(
        (matrix_df[count_col], (row, col)),
        shape=(sub_c.categories.size, cha_c.categories.size),
    )

    return sparse_matrix, cha_c.categories


def plot_embed(
    embed_matrix: np.ndarray,
    vocabulary: Iterable[str],
    channel_name_df: pd.DataFrame,
    sub_name_df: pd.DataFrame,
    channel_topic_df: pd.DataFrame,
    color_df: Optional[pd.DataFrame] = None,
    n_jobs: int = 10,
    color_continuous_scale=px.colors.sequential.Viridis,
    color_continuous_midpoint=None,
    range_color=None,
):
    """Plot 2D UMAP embedding for provided matrix"""

    # apply umap
    fit = umap.UMAP(n_jobs=10)
    u = fit.fit_transform(embed_matrix)

    # create dataframe
    res_df = pd.DataFrame(u, columns=["x", "y"])
    res_df["vocab"] = vocabulary

    # merge with name dataframes
    name_df = (
        res_df.merge(channel_name_df, left_on="vocab", right_on="channelId", how="left")
        .merge(sub_name_df, left_on="vocab", right_on="subreddit_id", how="left")
        .merge(channel_topic_df, on="channelId", how="left")
    )

    if color_df is not None:
        name_df = name_df.merge(color_df, on="channelTitle", how="left")

    # label is either reddit or youtube topic
    name_df["label"] = "reddit"
    name_df.loc[name_df["channelId"].notna(), "label"] = name_df.loc[
        name_df["channelId"].notna()
    ]["topic"]
    name_df["fullname"] = name_df["subreddit"].fillna("") + name_df[
        "channelTitle"
    ].fillna("")

    # plot
    fig = px.scatter(
        name_df,
        x="x",
        y="y",
        hover_data=["fullname"],
        color="label" if color_df is None else "color",
        color_continuous_scale=color_continuous_scale,
        color_continuous_midpoint=color_continuous_midpoint,
        range_color=range_color,
    )
    fig.show("notebook")

    return name_df, fig, fit
