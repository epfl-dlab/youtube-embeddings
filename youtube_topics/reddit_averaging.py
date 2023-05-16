from typing import Any, Dict, Iterable, Optional
import ipywidgets as widgets
import plotly.io as pio
import numpy as np
import plotly.express as px
import umap
import pandas as pd
import itertools



import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

import pandas as pd
import scipy
import yaml
import plotly.express as px
import fbpca
import umap
import matplotlib.patheffects as PathEffects
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import fbpca

from youtube_topics.directions import DimenGenerator
from youtube_topics.embeddings import csr_from_df, plot_embed
from youtube_topics.pmi import calculate_pmi


from sklearn.neighbors import NearestNeighbors

pio.renderers.default = "notebook"

from ipywidgets import interact_manual
import ipywidgets as widgets
import seaborn as sns


def read_seed_dim_dict(path):
    """Read yaml file with seeds

    The seeds for a dimension are pairs of subreddits (from left and right side of dimension)
    """
    with open(path, "r") as stream:
        dimensions = yaml.safe_load(stream)["dimensions"]

        seed_dim_dict = {x["name"]: pd.DataFrame(x["seeds"]) for x in dimensions}

    return seed_dim_dict


def normalize_matrix(matrix):
    """Normalize scipy sparse matrix using multiplication with diagonal matrix of sum(axis=1)"""
    x = np.array(1 / matrix.sum(axis=1)).reshape(-1)
    y = scipy.sparse.spdiags(x, 0, x.size, x.size)
    return y.T @ matrix


def compute_vecs(matrix, embed_vec, categorical_sub):
    """Compute per channel vectors by doing dot products with subreddit embedding"""
    return normalize_matrix(matrix) @ embed_vec.loc[categorical_sub.categories]


def compute_shrink_any(matrix, embed_vec, categorical_sub, aggregator, k=0):
    """Compute shrinkage embedding"""

    agg = aggregator(matrix)

    m_avg = agg.mean() + k
    m_i = agg

    vecs = compute_vecs(matrix, embed_vec, categorical_sub)

    return np.multiply(vecs, (m_i / (m_avg + m_i))) + np.multiply(
        (m_avg / (m_avg + m_i)), vecs.mean(axis=0)[None, :]
    )


def compute_shrink(matrix, embed_vec, categorical_sub, k=0):
    """Compute shrinkage embedding"""
    aggregator = lambda matrix: matrix.sum(axis=1)
    return compute_shrink_any(matrix, embed_vec, categorical_sub, aggregator, k=k)


def get_dim_unnormed(seed_dim_dict, seed, channel_vectors, subreddit_vectors):
    channel_vectors_norm = channel_vectors.divide(
        np.linalg.norm(channel_vectors.values, axis=1), axis="rows"
    )
    subreddit_vectors_norm = subreddit_vectors.divide(
        np.linalg.norm(subreddit_vectors.values, axis=1), axis="rows"
    )

    return get_vects_from_seed(
        seed_dim_dict, seed, channel_vectors_norm, subreddit_vectors_norm
    )


def get_vects_from_seed(
    seed_dim_dict, seed, channel_vectors_norm, subreddit_vectors_norm
):
    """Computes dimension for one particular seed, based on seeds in seed_dim_dict

    Requires normalized channel and subreddit vectors
    """

    # get seed, filter all pairs such that both subreddits are in our dataset
    seed_df = seed_dim_dict[seed]

    # compute difference vector
    seed_vect = (
        subreddit_vectors_norm.loc[seed_df[1]].values
        - subreddit_vectors_norm.loc[seed_df[0]].values
    ).mean(axis=0)

    ness_vect = (
        subreddit_vectors_norm.loc[seed_df[1]].values
        + subreddit_vectors_norm.loc[seed_df[0]].values
    ).mean(axis=0)

    # compute scores
    sub_score = subreddit_vectors_norm @ (seed_vect / np.linalg.norm(seed_vect))
    channel_score = channel_vectors_norm @ (seed_vect / np.linalg.norm(seed_vect))

    sub_ness = subreddit_vectors_norm @ (ness_vect / np.linalg.norm(ness_vect))
    channel_ness = channel_vectors_norm @ (ness_vect / np.linalg.norm(ness_vect))

    return sub_score, sub_ness, channel_score, channel_ness


def get_scores_df(seed_dim_dict, channel_vects, subreddit_vects):
    """Create a dataframe with dimensions for all seeds in seed_dim_dict

    Returns such a dataframe for youtube channels and subreddits
    """

    channel_vectors_norm = channel_vects.divide(
        np.linalg.norm(channel_vects.values, axis=1), axis="rows"
    )
    subreddit_vectors_norm = subreddit_vects.divide(
        np.linalg.norm(subreddit_vects.values, axis=1), axis="rows"
    )

    seed_vec_dict = {
        seed: get_vects_from_seed(
            seed_dim_dict, seed, channel_vectors_norm, subreddit_vectors_norm
        )
        for seed in seed_dim_dict
    }

    sub_score_df = pd.concat(
        [v[0].to_frame(f"{k}") for k, v in seed_vec_dict.items()], axis=1
    )
    sub_ness_df = pd.concat(
        [v[1].to_frame(f"{k}-ness") for k, v in seed_vec_dict.items()], axis=1
    )

    chan_score_df = pd.concat(
        [v[2].to_frame(f"{k}") for k, v in seed_vec_dict.items()], axis=1
    )
    chan_ness_df = pd.concat(
        [v[3].to_frame(f"{k}-ness") for k, v in seed_vec_dict.items()], axis=1
    )

    full_sub_df = pd.concat((sub_score_df, sub_ness_df), axis=1)
    full_chan_df = pd.concat((chan_score_df, chan_ness_df), axis=1)

    return full_sub_df, full_chan_df


def averaging_embed(
    cooc_df: pd.DataFrame,
    embed_vec: pd.DataFrame,
    pmi: bool = False,
    shrinkage: bool = False,
    item_col: str = "channelId",
    sub_col: str = "subreddit",
    count_col: str = "total",
    common_subreddits: Optional[Iterable[str]] = None,
    pmi_kwargs: Dict[Any, Any] = None,
) -> pd.DataFrame:
    """Create embedding using subreddit coocurrence

    Can use channel-level or video-level coocs, by changing item_col
    If common_subreddits is provided, must be sorted

    Args:
        cooc_df (pd.DataFrame): Cooccurrence df
        embed_vec (pd.DataFrame): Subreddit embeddings
        pmi (bool, optional): Whether to apply PMI to coocs. Defaults to False.
        shrinkage (bool, optional): Whether to apply bayesian shrinkage. Defaults to False.
        item_col (str, optional): Item column (either channel or video). Defaults to "channelId".
        sub_col (str, optional): Subreddit column. Defaults to "subreddit".
        common_subreddits (Optional[Iterable[str]], optional): If provided, computes embeds for a subset of subreddits. Defaults to None.
        pmi_kwargs (Dict[Any, Any], optional): Dictionary of PMI arguments (shift_type, threshold, alpha). Defaults to None.

    Raises:
        ValueError: If common_subreddits is provided but not sorted

    Returns:
        pd.DataFrame: Item embeddings
    """

    if common_subreddits is not None:
        if list(common_subreddits) != sorted(common_subreddits):
            raise ValueError("common_subreddits must be sorted")
    else:
        common_subreddits = sorted(cooc_df[sub_col].unique())

    # default pmi kwargs
    if pmi_kwargs is None:
        pmi_kwargs = dict(shift_type=0, threshold=0, alpha=0.5)

    # pd categories
    categorical_sub = pd.CategoricalDtype(common_subreddits, ordered=True)
    cha_c = pd.CategoricalDtype(sorted(cooc_df[item_col].unique()), ordered=True)

    # construct scipy csr matrix from cooc
    csr, _ = csr_from_df(
        cooc_df,
        sub_c=categorical_sub,
        cha_c=cha_c,
        count_col=count_col,
        subreddit_col=sub_col,
        channel_col=item_col,
    )

    # compute pmi
    if pmi:
        cooc_matrix = calculate_pmi(csr.T.astype(np.float64), **pmi_kwargs)
    else:
        cooc_matrix = csr.T

    to_df = lambda vecs: pd.DataFrame(vecs, index=cha_c.categories)

    # compute shrinkage
    if shrinkage:
        return to_df(compute_shrink(cooc_matrix, embed_vec, categorical_sub))
    else:
        return to_df(compute_vecs(cooc_matrix, embed_vec, categorical_sub))


def embed_stratify(
    video_embed, channel_vid_cooc, channels, weight_func="log", shrink=True
):
    """Helper function used by video_stratify

    Computes channel embed from video embed and channel/video coocurrence matrix"""

    assert isinstance(channel_vid_cooc, scipy.sparse._csc.csc_matrix)

    def log_weight(chan_cooc):
        """Log weight, warning: might modify original matrix"""
        log_matrix = chan_cooc.copy()
        log_matrix.data = np.log(log_matrix.data + 1)
        return normalize_matrix(log_matrix)

    def equal_weight(chan_cooc):
        nonzero_cooc = chan_cooc != 0
        return normalize_matrix(nonzero_cooc)

    weight_func_dict = {"log": log_weight, "equal": equal_weight}

    # cooccurrence function
    assert weight_func in weight_func_dict
    weight_chan = weight_func_dict[weight_func](channel_vid_cooc)

    # actual vectors
    dotted = weight_chan @ video_embed

    if shrink:
        # compute shrinkage per channel
        num_vids = (channel_vid_cooc != 0).sum(axis=1)
        m_avg = num_vids.mean()
        m_i = num_vids
        shrink_matrix = np.multiply(
            dotted,
            (m_i / (m_avg + m_i))
            + np.multiply((m_avg / (m_avg + m_i)), dotted.mean(axis=0)),
        )

        return pd.DataFrame(shrink_matrix, index=channels)

    return pd.DataFrame(dotted, index=channels)


def video_stratify(
    cooc_df: pd.DataFrame,
    embed_vec: pd.DataFrame,
    pmi: bool = False,
    shrinkage_channel: bool = False,
    shrinkage_video: bool = False,
    video_col: str = "vid_id",
    chan_col: str = "channelId",
    sub_col: str = "subreddit",
    count_col: str = "total",
    weight_func: str = "log",
    common_subreddits: Optional[Iterable[str]] = None,
    pmi_kwargs: Dict[Any, Any] = None,
) -> pd.DataFrame:
    """Compute channel embedding by using the video stratification method

    Args:
        cooc_df (pd.DataFrame): Video, Channel, Subreddit cooccurrence df
        embed_vec (pd.DataFrame): Subreddit embeddings
        pmi (bool, optional): Whether to use PMI. Defaults to False.
        shrinkage_channel (bool, optional): Whether to use shrinkage on a channel level. Defaults to False.
        shrinkage_video (bool, optional): Whether to use shrinkage on a video level. Defaults to False.
        video_col (str, optional): Column in cooc_dffor video identifier. Defaults to "vid_id".
        chan_col (str, optional): Column in cooc_df for channel identifier. Defaults to "channelId".
        sub_col (str, optional): Column in cooc_df for subreddit. Defaults to "subreddit".
        weight_func (str, optional): Either 'log', or 'equal', weighting for videos in channels. Defaults to 'log'.
        common_subreddits (Optional[Iterable[str]], optional):  If provided, computes embeds for a subset of subreddits. Defaults to None.
        pmi_kwargs (Dict[Any, Any], optional): Dictionary of PMI arguments (shift_type, threshold, alpha). Defaults to None.

    Returns:
        pd.DataFrame: Channel embeddings
    """

    # compute embedding per video
    video_embed = averaging_embed(
        cooc_df,
        embed_vec,
        pmi=pmi,
        shrinkage=shrinkage_video,
        item_col=video_col,
        sub_col=sub_col,
        count_col=count_col,
        common_subreddits=common_subreddits,
        pmi_kwargs=pmi_kwargs,
    )

    # compute cooccurrences between videos and channels
    vid_c = pd.CategoricalDtype(sorted(cooc_df[video_col].unique()), ordered=True)
    cha_c = pd.CategoricalDtype(sorted(cooc_df[chan_col].unique()), ordered=True)
    channel_vid_cooc, _ = csr_from_df(
        cooc_df,
        sub_c=vid_c,
        cha_c=cha_c,
        count_col=count_col,
        subreddit_col=video_col,
        channel_col=chan_col,
    )
    channel_vid_cooc = channel_vid_cooc.T

    # compute channel embeddings using embeddings per video and channel / video cooccurrences
    return embed_stratify(
        video_embed,
        channel_vid_cooc,
        cha_c.categories,
        shrink=shrinkage_channel,
        weight_func=weight_func,
    )
