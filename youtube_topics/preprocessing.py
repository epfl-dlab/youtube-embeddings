from typing import List
import pandas as pd
from IPython.core.display_functions import display


def filter_spammy_authors(df, max_posts=1_000, author_col="author", keep_deleted=True):
    """Removes authors which have more than max_posts

    Args:
        df (pd.DataFrame): dataframe to filter
        max_posts (int, optional): maximum posts per author. Defaults to 1_000.
        author_col (str, optional): col of author. Defaults to 'author'.
        keep_deleted (bool, optional): whether to remove posts from deleted authors. Defaults to True.

    Returns:
        pd.DataFrame: filtered dataframe
    """

    per_author = df.groupby(author_col).size()

    remove_authors = set(per_author.loc[lambda x: x > max_posts].index)

    if keep_deleted:
        remove_authors = remove_authors - set(["[removed]"])

    return df[~df[author_col].isin(remove_authors)]


gpby_counts = (
    lambda df: df.groupby(["subreddit", "channelId"])
    .size()
    .to_frame("total")
    .reset_index()
)


def collect_entities(
    ids_df: pd.DataFrame,
    channel_col: str = "channelId",
    sub_col: str = "subreddit_id",
    video_col: str = "vid_id",
):
    """Computes unique #channels, #subreddits and #videos in a dataframe

    Helper method used by filter_all
    """

    unique_chans = ids_df[channel_col].nunique()
    unique_subs = ids_df[sub_col].nunique()

    if video_col in ids_df.columns:
        unique_vids = ids_df[video_col].nunique()
    else:
        unique_vids = 0

    return len(ids_df), unique_chans, unique_subs, unique_vids


def filter_all(
    ids_df: pd.DataFrame,
    sub_col: str = "subreddit_id",
    channel_col: str = "channelId",
    video_col: str = "vid_id",
    promotion_df_path: str = "promotion_subreddits.csv",
    additional_removals: List[str] = ["t5_hhj1w", "t5_2ub5q"],
    num_videos: int = 2,
    num_subreddits: int = 5,
    threshold: int = 1_000,
    channel_info_path: str = "channel_info.jsonl.gz",
    max_iters: int = 20,
    chan_per_sub: int = 5,
    sub_per_chan: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """Applies multiple filters

    - Removes all subreddit which are meant for promoting new channels
    - Filters channels which have "num_videos" or more videos shared over "num_subreddits" or more subreddits
    - Filters channels which have more than "threshold" subscribers
    - Filters channels which are shared over more than "sub_per_chan" subreddits
    And subreddits which share more than "chan_per_sub" channels.

    Args:
        ids_df (pd.DataFrame): Original (vid_id, chan_id, sub_id) DataFrame
        sub_col (str, optional): Subreddit id column. Defaults to "subreddit_id".
        channel_col (str, optional): Column of channel id. Defaults to "channelId".
        video_col (str, optional): Column of video id. Defaults to "vid_id".
        promotion_df_path (str, optional): Path to saved promotion dataframe. Defaults to "promotion_subreddits.csv".
        additional_removals (List[str], optional):  Additional subreddit ids to remove. Defaults to ["t5_hhj1w", "t5_2ub5q"].
        num_videos (int, optional): Threshold for number of videos. Defaults to 2.
        num_subreddits (int, optional): Threshold for number of subreddits. Defaults to 5.
        threshold (int, optional): Number of channels threshold. Defaults to 1_000.
        channel_info_path (str, optional): Path to channel info dataframe. Defaults to "channel_info.jsonl.gz".
        max_iters (int, optional): Max number of iterations. Defaults to 20.
        chan_per_sub (int, optional): Threshold on subs. Defaults to 10.
        sub_per_chan (int, optional): Threshold on channels. Defaults to 10.

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if verbose:
        steps_lens = []
        print("Removing all subreddit which are meant for promoting new channels\n")
        steps_lens.append(
            collect_entities(
                ids_df, video_col=video_col, channel_col=channel_col, sub_col=sub_col
            )
        )

    ids_df = filter_promotion_subreddit(
        ids_df,
        promotion_df_path=promotion_df_path,
        additional_removals=additional_removals,
        sub_col=sub_col,
    )

    if verbose:
        print(
            'Filtering channels which have "num_videos" or more videos shared over "num_subreddits" or more subreddits\n'
        )
        steps_lens.append(
            collect_entities(
                ids_df, video_col=video_col, channel_col=channel_col, sub_col=sub_col
            )
        )

    ids_df = filter_enough_videos(
        ids_df,
        num_videos=num_videos,
        num_subreddits=num_subreddits,
        video_col=video_col,
        channel_col=channel_col,
        sub_col=sub_col,
    )

    if verbose:
        print('Filtering channels which have more than "threshold" subscribers\n')
        steps_lens.append(
            collect_entities(
                ids_df, video_col=video_col, channel_col=channel_col, sub_col=sub_col
            )
        )

    ids_df = filter_num_subscribers(
        ids_df,
        threshold=threshold,
        channel_info_path=channel_info_path,
        channel_col=channel_col,
    )

    if verbose:
        print(
            'Filtering channels which are shared over more than "sub_per_chan" subreddits And subreddits which share more than "chan_per_sub" channels.\n'
        )
        steps_lens.append(
            collect_entities(
                ids_df, video_col=video_col, channel_col=channel_col, sub_col=sub_col
            )
        )

    ids_df = filter_co_occurrences(
        ids_df,
        max_iters=max_iters,
        chan_per_sub=chan_per_sub,
        sub_per_chan=sub_per_chan,
        sub_col=sub_col,
        channel_col=channel_col,
    )

    if verbose:
        steps_lens.append(
            collect_entities(
                ids_df, video_col=video_col, channel_col=channel_col, sub_col=sub_col
            )
        )
        display(
            pd.DataFrame(
                steps_lens,
                columns=["#rows", "#channels", "#subreddits", "#videos"],
                index=["start", "promote", "enough_vids", "subscribers", "co-oc"],
            ).applymap(lambda x: f"{x:,}")
        )

    return ids_df


def filter_promotion_subreddit(
    ids_df: pd.DataFrame,
    promotion_df_path: str = "promotion_subreddits.csv",
    additional_removals: List[str] = ["t5_hhj1w", "t5_2ub5q"],
    sub_col: str = "subreddit_id",
) -> pd.DataFrame:
    """Removes all subreddit which are meant for promoting new channels

    Args:
        ids_df (pd.DataFrame): Original (vid_id, chan_id, sub_id) DataFrame
        promotion_df_path (str, optional): Path to saved promotion dataframe. Defaults to "promotion_subreddits.csv".
        additional_removals (List[str], optional): Additional subreddit ids to remove. Defaults to ["t5_hhj1w", "t5_2ub5q"].
        sub_col (str, optional): Subreddit id column. Defaults to "subreddit_id".

    Returns:
        pd.DataFrame: Filtered DataFrame
    """

    promotion_subs_df = pd.read_csv(promotion_df_path)
    promotion_subs = list(promotion_subs_df[promotion_subs_df["promotion"]][sub_col])

    return ids_df[~ids_df[sub_col].isin(promotion_subs + additional_removals)]


def filter_enough_videos(
    ids_df: pd.DataFrame,
    num_videos: int = 2,
    num_subreddits: int = 5,
    video_col: str = "vid_id",
    channel_col: str = "channelId",
    sub_col: str = "subreddit_id",
) -> pd.DataFrame:
    """Filters channels which have "num_videos" or more videos shared over "num_subreddits" or more subreddits

    Args:
        ids_df (pd.DataFrame): Original (vid_id, chan_id, sub_id) DataFrame
        num_videos (int, optional): Threshold for number of videos. Defaults to 2.
        num_subreddits (int, optional): Threshold for number of subreddits. Defaults to 5.
        video_col (str, optional): Column of video id. Defaults to "vid_id".
        channel_col (str, optional): Column of channel id. Defaults to "channelId".
        sub_col (str, optional): Column of subreddit id. Defaults to "subreddit_id".

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    ids_df = ids_df.astype("category")

    # filter videos which are shared over more than "num_subreddits" subreddits
    subs_per_vid = (
        ids_df.groupby(video_col)[sub_col]
        .nunique()
        .loc[lambda x: x >= num_subreddits]
        .to_frame("total")
    )

    # filter channels from previous videos
    filter_vids = (
        ids_df[[video_col, channel_col]]
        .drop_duplicates()
        .set_index(video_col)
        .join(subs_per_vid, how="inner")
    )

    # channels which have more than "num_videos" left
    channels_filtered = (
        filter_vids.reset_index()
        .groupby(channel_col)[video_col]
        .nunique()
        .loc[lambda x: x >= num_videos]
        .index
    )

    return ids_df[ids_df[channel_col].isin(channels_filtered)]


def filter_num_subscribers(
    ids_df: pd.DataFrame,
    threshold: int = 1_000,
    channel_info_path: str = "channel_info.jsonl.gz",
    channel_col: str = "channelId",
) -> pd.DataFrame:
    """Filters channels which have more than "threshold" subscribers

    Args:
        ids_df (pd.DataFrame): Original (vid_id, chan_id, sub_id) DataFrame
        threshold (int, optional): Number of channels threshold. Defaults to 1_000.
        channel_info_path (str, optional): Path to channel info dataframe. Defaults to "channel_info.jsonl.gz".
        channel_col (str, optional): Column of channel id. Defaults to "channelId".

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    channel_info_df = pd.read_json(channel_info_path, lines=True)

    ids_df = ids_df.merge(
        channel_info_df[[channel_col, "subscriberCount"]].rename(
            columns={"id": channel_col}
        ),
        on=channel_col,
        how="inner",
    )

    return ids_df[ids_df["subscriberCount"] > threshold]


def filter_co_occurrences(
    ids_df: pd.DataFrame,
    max_iters: int = 20,
    chan_per_sub: int = 10,
    sub_per_chan: int = 10,
    sub_col: str = "subreddit_id",
    channel_col: str = "channelId",
) -> pd.DataFrame:
    """Filters channels which are shared over more than "sub_per_chan" subreddits
    And subreddits which share more than "chan_per_sub" channels.

    Args:
        ids_df (pd.DataFrame): Original (vid_id, chan_id, sub_id) DataFrame
        max_iters (int, optional): Max number of iterations. Defaults to 20.
        chan_per_sub (int, optional): Threshold on subs. Defaults to 10.
        sub_per_chan (int, optional): Threshold on channels. Defaults to 10.
        sub_col (str, optional): Column for subreddit id. Defaults to "subreddit_id".
        channel_col (str, optional): Column for channel id. Defaults to "channelId".

    Returns:
        pd.DataFrame: Filtered DataFrame
    """

    matrix_df = (
        ids_df.astype(str)
        .groupby([sub_col, channel_col])
        .size()
        .to_frame("total")
        .reset_index()
    )

    matrix_len = len(matrix_df)
    len_changed = True

    # Filter iteratively until no change or max iters reached
    while len_changed and max_iters > 0:
        # get number of subreddits per channel, channels per subreddit
        unique_subreddits_per_channel = (
            matrix_df.groupby(channel_col)[sub_col].apply(set).apply(len)
        )
        unique_channels_per_subreddit = (
            matrix_df.groupby(sub_col)[channel_col].apply(set).apply(len)
        )

        # filter channels and subreddit
        matrix_df = matrix_df.merge(
            unique_channels_per_subreddit.loc[lambda x: x > chan_per_sub].reset_index()[
                [sub_col]
            ],
            on=sub_col,
            how="inner",
        ).merge(
            unique_subreddits_per_channel.loc[lambda x: x > sub_per_chan].reset_index()[
                [channel_col]
            ],
            on=channel_col,
            how="inner",
        )

        # check if we continue to iterate
        if len(matrix_df) != matrix_len:
            len_changed = True
            matrix_len = len(matrix_df)
        else:
            len_changed = False

        max_iters -= 1

    return matrix_df
