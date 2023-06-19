import base64
import json
from itertools import combinations
import random
import pandas as pd
from tqdm.auto import tqdm
import scipy
import jmespath
import logging
import numpy as np
import gspread
import innertube
import functools

from functools import lru_cache
from .inner_proxies.channel import time_to_sec
from .inner_proxies.multiprocessing import retry

from statsmodels.stats.inter_rater import fleiss_kappa


def df_to_sheet(gc, df, sheetname, client=None):
    """Upload dataframe of youtube channels to google sheet

    Args:
        gc: gspread instance
        df: dataframe of youtube channels
        sheetname: name of sheet to upload to
        client: innertube client. Defaults to None.
    """

    if client is None:
        client = innertube.InnerTube("WEB")

    df = df.reset_index(drop=True)

    get_channels_summary = lambda col: df[col].apply(
        lambda chanid: recent_thumbnails(client, chanid, nthumbs=4)
        .head(4)
        .to_dict(orient="records")
    )

    process_title = (
        lambda s, col: s.apply(lambda l: [x["title"] for x in l])
        .apply(pd.Series)
        .rename(columns=lambda i: f"{col}_{i}")
    )
    process_thumb = (
        lambda s, col: s.apply(lambda l: [x["thumbnail"] for x in l])
        .apply(pd.Series)
        .rename(columns=lambda i: f"{col}_{i}")
        .applymap(lambda x: f'=IMAGE("{x}")')
    )

    channel_summaries = [(col, get_channels_summary(col)) for col in tqdm(df.columns)]

    df_thumb = pd.concat(
        [df] + [process_thumb(summ, col) for col, summ in channel_summaries], axis=1
    )
    df_title = pd.concat(
        [df] + [process_title(summ, col) for col, summ in channel_summaries], axis=1
    )

    fulldf = pd.concat((df_thumb, df_title)).sort_index(kind="mergesort")

    sh = gc.open(sheetname)
    sh.sheet1.update(
        [fulldf.columns.values.tolist()] + fulldf.values.tolist(),
        value_input_option="USER_ENTERED",
    )


def prep_mturk_batch(mturkified, batch_size=10):
    """Create batches from dataframe of comparisons

    Transforms batches into json, b64 encoded, which will get read by
    our js script on mturk.
    """

    # turn all rows to dics (+ list outside so we can sum)
    row_to_dics = (
        mturkified.apply(lambda s: [s.to_dict()], axis=1)
        .to_frame("jsons")
        .reset_index()
    )

    # sum our lists to concat for those in same batch
    grouped_batches = (
        row_to_dics.assign(mod=row_to_dics["index"] // batch_size)
        .groupby("mod")[["jsons"]]
        .sum()
    )

    # encode everything in b64
    b64_encoded = grouped_batches.assign(
        length=grouped_batches.jsons.apply(len),
        jsons=grouped_batches.jsons.apply(
            lambda s: base64.b64encode(json.dumps(s).encode()).decode()
        ),
    )

    return b64_encoded


def embed_closest(embed):
    """Create function to get the pair of channel which is the closest with respect to provided embedding"""

    def apply_closest(row):
        vals = combinations(row.values, r=2)
        cols = ["".join(x) for x in combinations(row.index, r=2)]
        return max(
            ((col, embed.loc[a] @ embed.loc[b]) for col, (a, b) in zip(cols, vals)),
            key=lambda x: x[1],
        )[0]

    return apply_closest


def embed_furthest(embed):
    """Create function to get the pair of channel which is the furthest with respect to provided embedding"""

    def apply_closest(row):
        vals = combinations(row.values, r=2)
        cols = ["".join(x) for x in combinations(row.index, r=2)]
        return min(
            ((col, embed.loc[a] @ embed.loc[b]) for col, (a, b) in zip(cols, vals)),
            key=lambda x: x[1],
        )[0]

    return apply_closest


def sample_similar(sample, neigh, embed, frac=0.05):
    """Get the most similar channel, and nth channel away from sampled channel"""

    neighs = neigh.kneighbors(
        embed.loc[[sample]],
        n_neighbors=int(embed.shape[0] * frac),
        return_distance=False,
    ).ravel()

    ind = embed.index[neighs]
    ind = ind[ind != sample]

    return [ind[0], ind[-1]]


def input_expected_result(b64_json, embed):
    original_df = pd.DataFrame(json.loads(base64.b64decode(b64_json)))
    tmp_df = original_df.rename(
        columns={"A_channelId": "A", "B_channelId": "B", "C_channelId": "C"}
    )[["A", "B", "C"]]

    rev_dict = {"AB": "C", "BC": "A", "AC": "B"}

    def apply_closest(row):
        try:
            vals = combinations(row.values, r=2)
            cols = ["".join(x) for x in combinations(row.index, r=2)]
            return rev_dict[
                max(
                    (
                        (col, embed.loc[a] @ embed.loc[b])
                        for col, (a, b) in zip(cols, vals)
                    ),
                    key=lambda x: x[1],
                )[0]
            ]
        except KeyError:
            return "A"

    return tmp_df.apply(apply_closest, axis=1).tolist()


def sample_similar_mturk(
    embed, nn, frac, client, seed=0, num_samples=20, ignore_channels=None
):
    """Create sample dataframe from embedding, nearest neighbours and fraction"""

    # set seed
    random.seed(seed)

    # if we have to sample anything but channels in ignore_channels
    if ignore_channels is not None:
        sample_embed = embed.loc[~embed.index.isin(ignore_channels)]
    else:
        sample_embed = embed

    # sample a random channel
    chans = sample_embed.sample(num_samples, random_state=seed, replace=False).index

    expected = []
    shuffled = []

    for chan in tqdm(chans):
        # find nearest neighbour, frac*len(embed)th nearest neighbour
        neighs = [chan] + sample_similar(chan, nn, embed, frac=frac)

        # append them in order, and not in order
        expected.append(neighs)
        shuffled.append(random.sample(neighs, len(neighs)))

    # get the dataframes
    expdf = pd.DataFrame(expected, columns=["A", "B", "C"])
    shufdf = pd.DataFrame(shuffled, columns=["A", "B", "C"])
    mturked_df = mturkify(client, shufdf).reset_index(drop=True)

    return expdf, shufdf, mturked_df


def read_mturk_res(path, cols=["HITId"], filter_low=0, filter_up=float("inf")):
    """Read mturk csv results

    Args:
        path: path to csv
        cols: cols to keep. Defaults to ['HITId'].

    Returns:
        res df
    """
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "Input.frac": "frac",
            "Input.embed": "embed",
            "Input.mod": "mod",
            "Answer.batch-results": "res",
        }
    )[cols + ["res"]]
    df["res"] = df["res"].apply(lambda s: [x.upper() for x in s.split(",")])
    return df[(df.res.apply(len) >= filter_low) & (df.res.apply(len) <= filter_up)]


def agreement_per_hit(col):
    """Create function to compute agreement between workers on one particular hit"""

    def agg_per_hit(hitdf):
        result = (
            hitdf.explode(col).reset_index().rename(columns={"index": "col1_index"})
        )
        result["ind"] = result.groupby("col1_index").cumcount()

        cats = pd.CategoricalDtype(["A", "B"], ordered=True)
        result["polnum"] = result[col].astype(cats).cat.codes

        agree_df = (
            result.assign(val=1)
            .pivot_table(
                index=["HITId", "ind"], values="val", columns="polnum", aggfunc=np.sum
            )
            .fillna(0)
        )

        return fleiss_kappa(agree_df)

    return agg_per_hit


def agreement_number(col):
    """Compute maximum number of agreements

    For example, if 3 workers agree on A, 1 worker chose C, 1 worker chose B, returns 3
    With 5 workers, minimum number is 2, maximum number is 5
    """

    def agg_number(df):
        tmp_arr = np.stack(df[col]).astype(object)
        tmp_arr[tmp_arr == "A"] = 0
        tmp_arr[tmp_arr == "B"] = 1
        tmp_arr[tmp_arr == "C"] = 2
        tmp_arr = tmp_arr.astype(int)

        return scipy.stats.mode(tmp_arr, axis=0, keepdims=False).count

    return agg_number


def upload_thumbnails(client, channel_id, nthumbs=10):
    """Get thumbnails from uploads playlists"""

    res = client.next(playlist_id="UU" + channel_id[2:])

    results = jmespath.search(
        "contents.twoColumnWatchNextResults.playlist.playlist.contents[*].playlistPanelVideoRenderer"
        ".[title.simpleText,thumbnail.thumbnails[-1].url, videoId, lengthText.simpleText]",
        res,
    )

    return pd.DataFrame(results, columns=["title", "thumbnail", "videoId", "length"])


def ignore_short_vids(df):
    """Remove any video which is less than 62 seconds in length

    In reality, we can get shorts which are more than 62 seconds, but they are much less frequent (and not exactly shorts).
    """
    df_tmp = df.dropna().copy()
    df_tmp = df_tmp[(df_tmp["length"] != "UPCOMING") & (df_tmp["length"] != "PREMIERE")]
    df_tmp["length"] = df_tmp["length"].apply(time_to_sec)
    df_tmp = df_tmp.query("length > 62")
    df_tmp = df_tmp.drop(columns="length")

    return df_tmp


def recent_thumbnails(client, channel_id, nthumbs=10, ret_chan_name=False):
    """Get thumbnails from first playlist in channel homepage, or from uploads if not enough found"""

    NUM_VIDEOS_SHOWN = 5

    res = client.browse(channel_id)

    results = jmespath.search(
        (
            "contents.twoColumnBrowseResultsRenderer.tabs[0].tabRenderer.content.sectionListRenderer"
            ".contents[*].itemSectionRenderer.contents[0].shelfRenderer.content.horizontalListRenderer"
            ".items[*].gridVideoRenderer.[title.simpleText, thumbnail.thumbnails[-1].url, videoId, thumbnailOverlays[0].thumbnailOverlayTimeStatusRenderer.text.simpleText]"
        ),
        res,
    )

    has_results = (
        (results is not None)
        and (len(results) != 0)
        and (len(results[0]) >= min(NUM_VIDEOS_SHOWN, nthumbs))
    )

    # if we get results from first playlist of homepage, return them
    if has_results:
        df = pd.DataFrame(
            results[0], columns=["title", "thumbnail", "videoId", "length"]
        )
        df = ignore_short_vids(df)

    # if there was no results, or not enough, search in uploads playlist
    if not has_results or len(df) <= nthumbs:
        df_up = upload_thumbnails(client, channel_id, nthumbs=100)
        df = ignore_short_vids(df_up)

        # no vids, return empty df
        if len(df) == 0:
            df = pd.DataFrame(
                {"title": [np.NaN], "thumbnail": [np.NaN], "videoId": [np.NaN]}
            )
            if ret_chan_name:
                return df, np.NaN
            return df

        # if we still dont have enough, just repeat existing channels
        while len(df) <= nthumbs:
            df = pd.concat((df, df))

    # remove length column after having filtered shorts
    if "length" in df.columns:
        df = df.drop(columns="length")

    # also return channel name
    if ret_chan_name:
        chan_name = jmespath.search("header.c4TabbedHeaderRenderer.title", res)
        return df, chan_name

    return df

class BlackBox:
    """All BlackBoxes are the same."""
    def __init__(self, contents):
        # TODO: use a weak reference for contents
        self._contents = contents

    @property
    def contents(self):
        return self._contents

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(type(self))

@retry(tries=5, delay=10)
@functools.lru_cache()
def _channel_df(blackbox_client, channel_id, nthumbs=4):
    client = blackbox_client.contents
    df, chan_name = recent_thumbnails(
        client, channel_id, ret_chan_name=True, nthumbs=nthumbs
    )

    df = df.head(nthumbs)

    df["videoId"] = df["videoId"].apply(lambda i: f"https://youtube.com/watch?v={i}")

    cols = df.columns
    concat_df = pd.concat(
        (
            pd.Series((x[col] for x in df.to_dict(orient="records")))
            .to_frame()
            .rename(lambda i: f"{col}_{i}")
            .T
            for col in cols
        ),
        axis=1,
    )

    return concat_df.assign(channelTitle=chan_name, channelId=channel_id)

def channel_df(client, channel_id, nthumbs=4):
    """Create DataFrame with metadata for uploading as csv to mturk"""
    return _channel_df(BlackBox(client),  channel_id, nthumbs=nthumbs)
    

def replace_emoji_characters(s):
    """Placeholder, no longer need to replace emoji characters"""
    return s


def channel_df_missing(client):
    """Get channel DataFrame, and log if we couldn't find any video for one channel"""

    def channel_df_missing_(chan):
        try:
            return channel_df(client, chan, nthumbs=4)

        except ValueError:
            logging.warning(f"{chan} has no videos")
            return pd.DataFrame()

    return channel_df_missing_


def mturkify(client, df):
    """Prepare DataFrame of channel Ids for turk batching

    Gets video titles, thumbnails, channel name for each column of dataframe
    """

    df = df.reset_index(drop=True)

    return (
        pd.concat(
            (
                pd.concat(list(df[col].apply(channel_df_missing(client)))).rename(
                    columns=lambda i: f"{col}_{i}"
                )
                for col in df.columns
            ),
            axis=1,
        )
        .fillna("@@@None@@@")
        .applymap(replace_emoji_characters)
    )
