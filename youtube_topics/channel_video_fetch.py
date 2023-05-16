import pandas as pd
import time
import logging
import numpy as np
import uuid
import traceback

from collections.abc import MutableMapping

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def fetch_upload_playlists(
    channels_fetch: pd.Series,
    api_key: str,
    query_categories="snippet,contentDetails,statistics",
):
    """[DEPR] Get youtube upload playlist from each channel in channels_fetch

    In reality this step can be skipped, the upload playlist is obtained by replacing 'UC' by 'UU' in channel Id beginning
    """

    CHUNK_SIZE = 50

    # build youtube service
    api_service_name = "youtube"
    api_version = "v3"
    youtube = build(api_service_name, api_version, developerKey=api_key)

    channels_dfs = []

    for i in tqdm(range(0, len(channels_fetch), CHUNK_SIZE)):
        # channel ids iterate
        channel_ids = ",".join(channels_fetch.iloc[i : i + CHUNK_SIZE])

        # build and execute request
        request = youtube.channels().list(
            part=query_categories, id=channel_ids, maxResults=50
        )
        response = request.execute()

        if response["pageInfo"]["totalResults"] > 0:
            channels_df = pd.DataFrame(response["items"])
            channels_df["playlistId"] = channels_df["contentDetails"].apply(
                lambda x: x["relatedPlaylists"]["uploads"]
            )
            channels_dfs.append(channels_df)

        time.sleep(1)

    fullchanneldf = pd.concat(channels_dfs)

    return fullchanneldf


def fetch_videos_playlists_full(
    playlists_fetch: pd.Series, api_key: str, query_categories="snippet,contentDetails"
):
    "Fetch latest videos from each playlist in playlists_fetch" ""

    playlist_dfs = []

    # build youtube service
    api_service_name = "youtube"
    api_version = "v3"
    youtube = build(api_service_name, api_version, developerKey=api_key)

    for playlistId in tqdm(playlists_fetch):
        try:
            request = youtube.playlistItems().list(
                part=query_categories, playlistId=playlistId, maxResults=50
            )
            response = request.execute()
        except HttpError as e:
            logging.warning(f"No videos found for {playlistId=}")

        if response["pageInfo"]["totalResults"] > 0:
            playlist_df = pd.DataFrame(response["items"])
            playlist_dfs.append(playlist_df)

    fullplaylist_df = pd.concat(playlist_dfs)
    return fullplaylist_df


def fetch_videos_playlists(
    playlists_fetch: pd.Series, api_key: str, query_categories="snippet,contentDetails"
):
    "Fetch latest videos from each playlist in playlists_fetch" ""

    playlist_dfs = []

    # build youtube service
    api_service_name = "youtube"
    api_version = "v3"
    youtube = build(api_service_name, api_version, developerKey=api_key)

    for playlistId in tqdm(playlists_fetch):
        try:
            request = youtube.playlistItems().list(
                part=query_categories, playlistId=playlistId, maxResults=50
            )
            response = request.execute()
        except HttpError as e:
            logging.warning(f"No videos found for {playlistId=}")

        if response["pageInfo"]["totalResults"] > 0:
            playlist_df = pd.DataFrame(response["items"])["snippet"].apply(pd.Series)
            playlist_dfs.append(playlist_df)

    fullplaylist_df = pd.concat(playlist_dfs)
    return fullplaylist_df


def get_videos_channels(unique_channels: pd.Series, api_key: str, n_jobs=20):
    """[DEPR] All in one function for fetching videos from channel

    Deprecated since fetch_upload_playlists is deprecated as well (getting playlist id is easy from channel id)
    """

    # get upload playlist for each channel
    splits = [pd.Series(x) for x in np.array_split(unique_channels.values, n_jobs)]
    all_dfs = Parallel(n_jobs=n_jobs)(
        delayed(fetch_upload_playlists)(split, api_key) for split in splits
    )
    channel_playlist_df = pd.concat(all_dfs).reset_index(drop=True)

    # get videos from each upload playlist
    playlist_splits = [
        pd.Series(x)
        for x in np.array_split(channel_playlist_df.playlistId.values, n_jobs)
    ]
    all_videos_dfs = Parallel(n_jobs=n_jobs)(
        delayed(fetch_videos_playlists)(split, api_key) for split in playlist_splits
    )
    videos_df = pd.concat(all_videos_dfs).reset_index(drop=True)

    return videos_df


def _flatten_dict_gen(d, parent_key, sep):
    """Helper for flatten_dict"""
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    """Helper for flatten_df, flattens dictionaries"""
    if pd.isna(d):
        return np.NaN
    return dict(_flatten_dict_gen(d, parent_key, sep))


def flatten_df(df, cols):
    """Flatten dataframe with nested columns

    One example is the snippet column from YouTube Data API, it will be split
    in columns such as snippet.publishedAt, snippet.channelId, etc..
    """

    # concat works if index is default only
    assert isinstance(df.index, pd.RangeIndex) and df.index.step == 1

    # initial, remove columns
    dfs_concat = [df.drop(columns=cols)]

    # build df from each column
    for col in cols:
        dfs_concat.append(
            pd.DataFrame.from_records(
                df[col].apply(flatten_dict).fillna("").apply(dict).tolist()
            ).rename(columns=lambda s: f"{col}.{s}")
        )

    # concatenate all dfs back together
    return pd.concat(dfs_concat, axis=1)


def video_metadata(vid_ids_df, path, api_key, chunksize=50):
    """Fetching video metadata from a series of video ids"""

    # create api service
    api_service_name = "youtube"
    api_version = "v3"
    youtube = build(api_service_name, api_version, developerKey=api_key)

    response_dfs = []

    for chunk_idx in tqdm(range(0, len(vid_ids_df), chunksize)):
        try:
            # chunk df
            tmp_df = vid_ids_df.iloc[chunk_idx : chunk_idx + chunksize]

            # list of vid ids to request
            vid_ids = ",".join(tmp_df)

            # create request
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics,topicDetails", id=vid_ids
            )
            response = request.execute()

            # go next if all videos were deleted
            if len(response["items"]) == 0:
                continue

            # get response as df
            response_df = pd.DataFrame(response["items"])

            # add to list
            response_dfs.append(response_df)

            time.sleep(1)

        except Exception as e:
            path = f"{str(uuid.uuid4())}.jsonl.gz"
            logging.warning(
                f"{traceback.format_exc()} : Saving problematic dataframe to {path}"
            )
            tmp_df.to_json(path, lines=True, orient="records")

    final_df = pd.concat(response_dfs).reset_index(drop=True)

    final_df.to_feather(path, compression="zstd")
