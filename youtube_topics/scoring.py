import pickle
import itertools
from typing import List
import pandas as pd
import numpy as np

from scipy.stats import kendalltau
from scipy.stats import bootstrap


def read_polarized_df(path):
    """Wrapper for pickle loading

    Args:
        path (str): path to read

    Returns:
        Any: loaded object
    """

    with open(path, "rb") as handle:
        return pickle.load(handle)


def kendalltau_score(
    dimension: pd.Series,
    polarized_df: pd.DataFrame,
    ordering: List[str],
    id_col: str = "channelId",
    bias_col: str = "bias",
):
    """Get kendall rank correlation between series and known polarized channels df

    Args:
        dimension (pd.Series): Series of dimensions to query
        polarized_df (pd.DataFrame): Known polarized channels df
        ordering (List[str]): Ordering of channel labels
        id_col (str, optional): Name of channel id column. Defaults to 'channelId'.
        bias_col (str, optional): Name of channel ordering label column. Defaults to 'bias'.

    Returns:
        float: kendall rank correllation
    """

    kendall_df = dimension.to_frame("dimension").join(
        polarized_df.set_index(id_col), how="inner"
    )

    ordering_dtype = pd.CategoricalDtype(ordering, ordered=True)

    kendall_df[bias_col] = kendall_df[bias_col].astype(ordering_dtype).cat.codes

    return kendalltau(
        kendall_df["dimension"], kendall_df["bias"], variant="c"
    ).correlation


def kendalltau_scorer(polarized_df, ordering, id_col="channelId", bias_col="bias"):
    """Get scoring function for computing kendall rank correlation given dimension series"""

    def kendalltau_scorer_(dimension):
        return {
            "kendall_tau": kendalltau_score(
                dimension, polarized_df, ordering, id_col=id_col, bias_col=bias_col
            )
        }

    return kendalltau_scorer_


def ordering_score(
    channel_scores,
    polarized_df,
    ordering,
    id_col="youtube_id",
    bias_col="bias",
    ret_ci=False,
    ret_full=False,
):
    """Deprecated, use kendall-tau instead"""

    all_pairs = []

    # generate all pairs
    for p1 in ordering:
        for p2 in ordering:
            if ordering.index(p1) < ordering.index(p2):
                all_pairs.append((p1, p2))

    chan_score_dict = (channel_scores - channel_scores.mean()).to_dict()

    pair_scores = []

    for p1, p2 in all_pairs:
        # df from pairs
        p1df = polarized_df[polarized_df[bias_col] == p1]
        p2df = polarized_df[polarized_df[bias_col] == p2]

        # combine them, get score
        all_pairs_df = pd.DataFrame(
            itertools.product(p1df[id_col], p2df[id_col]), columns=["left", "right"]
        )
        all_scores_df = all_pairs_df.applymap(chan_score_dict.get)

        # check if left_score < right_score
        orient_bool = (all_scores_df["left"] < all_scores_df["right"]).astype("float")
        mean = orient_bool.mean()

        # return all scores
        if ret_full:
            pair_scores.append((p1, p2, orient_bool))

        # return with ci
        elif ret_ci:
            from scipy.stats import bootstrap

            data = (orient_bool.astype("float"),)  # samples must be in a sequence
            ci = bootstrap(
                data, np.mean, confidence_level=0.95, random_state=1, batch=100
            ).confidence_interval
            pair_scores.append(p1, p2, mean, ci.low, ci.high)

        # return just mean
        else:
            pair_scores.append((p1, p2, mean))

    return pair_scores
