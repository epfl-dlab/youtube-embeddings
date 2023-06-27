---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Fetching video metadata using youtube data api v3

There's lots of information we are interested in for our big channels: the language, the ids from the most frequent videos, statistics.

We fetch all of this metadata from the youtube data api in this notebook

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

import logging
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed

from youtube_topics import data_path
from youtube_topics.channel_video_fetch import (fetch_upload_playlists,
                                                fetch_videos_playlists_full,
                                                flatten_df, video_metadata)
```

```python
logger = logging.getLogger()
logger.setLevel(logging.INFO)
```

### Read per channel dataset

```python
# chan_info is obtained by querying channel information using youtube data api on all channels obtained through
# reddit posts. we also mix in info from previous channels which have been deleted from youtube, and thus which is not
# available from youtube api
chan_info = pd.read_json(data_path("channel_info.jsonl.gz"), lines=True)
```

### Subset : more than 100K subscribers

```python
subscribers_threshold = (
    chan_info.query("subscriberCount > 100_000")
    .sort_values(by="subscriberCount", ascending=False)
    .reset_index(drop=True)
)
```

```python
subscribers_threshold.to_feather(
    data_path("filtered_channels.feather.zstd"), compression="zstd"
)
```

### Fetch upload playlist for each of those

```python
n_jobs = 20
api_key = "***REMOVED***"

unique_channels = subscribers_threshold["id"]
splits = [pd.Series(x) for x in np.array_split(unique_channels.values, n_jobs)]

all_dfs = Parallel(n_jobs=n_jobs)(
    delayed(fetch_upload_playlists)(
        split,
        api_key,
        query_categories="snippet,contentDetails,statistics,topicDetails,localizations",
    )
    for split in splits
)
channel_playlist_df = pd.concat(all_dfs).reset_index(drop=True)
```

#### Save the current df

```python
channel_playlist_df.to_feather(
    data_path("filtered_channels_default_playlist_df.feather.zstd"), compression="zstd"
)
```

#### Flatten df

All the data is saved in json / dictionary format. We first flatten those so everything can be accessed directly in the dataframe

```python
flattened_df = flatten_df(
    channel_playlist_df, ["snippet", "contentDetails", "statistics", "topicDetails"]
)
```

```python
flattened_df.to_feather(
    data_path("filtered_channels_default_playlist_flattened.feather.zstd"),
    compression="zstd",
)
```

### Fetch latest 50 videos for each channel (if possible) 

```python
# get videos from each upload playlist
playlist_splits = [
    pd.Series(x) for x in np.array_split(channel_playlist_df.playlistId.values, n_jobs)
]
all_videos_dfs = Parallel(n_jobs=n_jobs)(
    delayed(fetch_videos_playlists_full)(split, api_key) for split in playlist_splits
)
videos_df = pd.concat(all_videos_dfs).reset_index(drop=True)
```

```python
videos_df.to_feather(
    data_path("filtered_channels_videos.feather.zstd"), compression="zstd"
)
```

#### Flatten and save

```python
flattened_videos = flatten_df(videos_df, ["snippet", "contentDetails"])
flattened_videos = flattened_videos.drop_duplicates(
    subset=["snippet.channelId", "snippet.resourceId.videoId"]
)
flattened_videos = flattened_videos.reset_index(drop=True)
flattened_videos.to_feather(
    data_path("filtered_channels_videos_flattened.feather.zstd"), compression="zstd"
)
```

108 channels with no videos, otherwise got all of them.


### Cropping some columns from DF for smaller footprint

```python
videos = pd.read_feather(data_path("filtered_channels_videos_flattened.feather.zstd"))
renaming = {
    "snippet.publishedAt": "publishedAt",
    "snippet.channelId": "channelId",
    "snippet.title": "title",
    "snippet.description": "description",
    "snippet.resourceId.videoId": "videoId",
}
simplified_videos = videos[renaming.keys()].rename(columns=renaming)
simplified_videos.to_feather(
    data_path("filtered_channels_cropped.feather.zstd"), compression="zstd"
)
```

That filtered df is then used in the Semantic Embedding notebook.


### Fetching more data from each video

Unfortunately, fetching the video ids from each playlist is not enough, we need to fetch the `defaultAudioLanguage` , `defaultLanguage`, as well as the `snippet.categoryId` columns from each video.

```python
vid_ids = pd.read_feather(data_path("filtered_channels_videos_flattened.feather.zstd"))[
    "ids"
]

NUM_THREADS = 20
api_key = "***REMOVED***"

tmp_df_chunks = [
    vid_ids.iloc[x] for x in np.array_split(range(len(vid_ids)), NUM_THREADS)
]

# we did in multiple parts (1,2,3)
Parallel(n_jobs=NUM_THREADS)(
    delayed(video_metadata)(chunk, f"parts3/{i:02d}.feather.zstd")
    for i, chunk in enumerate(tmp_df_chunks)
)

# each part is flattened in a single df (here shown example for part 3)
flattened_df = flatten_df(
    pd.concat(
        pd.read_feather(path) for path in glob(data_path("parts3/*.feather.zstd"))
    ),
    ["snippet", "contentDetails", "statistics", "topicDetails"],
)
flattened_df.to_feather(data_path("flattened_video_info3.feather.zstd"))
```

### Reading back

```python
# we can then reread all parts :
flattened_paths = glob(data_path("flattened_video_info*.feather.zstd"))
vid_info = pl.concat([pl.read_ipc(path) for path in flattened_paths], how="diagonal")
vid_info = vid_info.unique(subset="id")
```

#### Save language df

```python
# save a filtered down df with only language info
vid_info[
    [
        "snippet.channelId",
        "snippet.defaultAudioLanguage",
        "snippet.defaultLanguage",
        "id",
    ]
].write_ipc("../data/per_vid_language.feather.zstd", compression="zstd")
```

#### Per channel category

```python
# aggregate by channel
agglist = vid_info.groupby("snippet.channelId").agg(
    [pl.col("snippet.categoryId").list(), pl.col("snippet.defaultAudioLanguage").list()]
)

# back to pandas
agglist_df = agglist.to_pandas()

# apply counts
agglist_df["snippet.categoryId"] = agglist_df["snippet.categoryId"].apply(Counter)

# rename
aggdf = agglist_df.rename(
    columns={
        "snippet.categoryId": "categoryId",
    }
)

# add major category, percentage of channels falling in that category
aggdf["major_cat"] = aggdf["categoryId"].apply(lambda c: c.most_common()[0][0])
aggdf["major_cat_percent"] = aggdf["categoryId"].apply(
    lambda c: c.most_common()[0][1] / sum(c.values())
)

# to save in feather
aggdf["categoryId"] = aggdf["categoryId"].apply(
    lambda c: {str(a): b for a, b in c.items()}
)

aggdf.to_feather(data_path("per_channel_aggs.feather.zstd"), compression="zstd")
```
