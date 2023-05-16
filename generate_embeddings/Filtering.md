---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Keeping only english speaking channels, recent uploads, intersection across embeddings

We take our large embeddings and filter down to 40K channels (the embeddings we share here)

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

from functools import reduce

import pandas as pd

from youtube_topics import data_path
```

#### Keep only english channels, recent content and at least 4 vids

```python
# filter only english speaking channels
newlang = pd.read_feather(
    data_path("old_files/channel_language_final_with_ignore.feather.zstd")
)
newl_en = newlang.query('label == "english"').set_index("channelId")[[]]

# other filters
videos_df = pd.read_feather(
    data_path("old_files/filtered_channels_videos_flattened.feather.zstd")
)
videos_df["video_dt"] = pd.to_datetime(videos_df["snippet.publishedAt"])

# recent > 2020
most_recent_vid = videos_df.groupby("snippet.channelId")["video_dt"].max()
recent_enough = most_recent_vid[
    most_recent_vid > pd.Timestamp(year=2020, month=1, day=1, tz="utc")
].to_frame("")[[]]

# at least 4 vids
per_chan_size = videos_df.groupby("snippet.channelId")[[]].size()
per_chan_min = per_chan_size.loc[lambda x: x >= 4].to_frame("a")[[]]

# read the embeds
recomm = pd.read_feather(
    data_path(
        "old_files/recomm64_undirected_walk40_context20_p4_q0_5_epochs_100.feather.zstd"
    )
).set_index("title")
reddit = pd.read_feather(data_path("old_files/reddit_vec.feather.zstd")).set_index(
    "channelId"
)
content = pd.read_feather(
    data_path("old_files/content_embedding.feather.zstd")
).set_index("channelId")

default_reddit = pd.read_feather(data_path("embeds/default.feather.zstd")).set_index(
    "channelId"
)

# join all indices
intersection_indices = sorted(
    reduce(
        set.intersection,
        (
            set(df.index)
            for df in [
                recomm,
                reddit,
                content,
                default_reddit,
                newl_en,
                recent_enough,
                per_chan_min,
            ]
        ),
    )
)

reddit_embed = reddit.loc[intersection_indices]
recomm_embed = recomm.loc[intersection_indices]
content_embed = content.loc[intersection_indices]
default_reddit = default_reddit.loc[intersection_indices]
```

### Save them

```python
# reddit_embed.rename_axis("channelId").reset_index().to_feather(
#     "../data/embeds/reddit.feather.zstd", compression="zstd"
# )
recomm_embed.rename_axis("channelId").reset_index().to_feather(
    data_path("embeds/recomm.feather.zstd"), compression="zstd"
)
content_embed.rename_axis("channelId").reset_index().to_feather(
    data_path("embeds/content.feather.zstd"), compression="zstd"
)
default_reddit.rename_axis("channelId").reset_index().to_feather(
    data_path("embeds/reddit.feather.zstd"), compression="zstd"
)
```

## Reddit dims

```python
reddit_embed = pd.read_feather(data_path("embeds/reddit.feather.zstd")).set_index(
    "channelId"
)

reddit_dims = pd.read_feather(data_path("dims/strat_log.feather.zstd")).set_index(
    "channelId"
)
default_dims = pd.read_feather(data_path("dims/default.feather.zstd")).set_index(
    "channelId"
)

reddit_dims = reddit_dims.join(reddit_embed[[]], how="inner")
default_dims = default_dims.join(reddit_embed[[]], how="inner")
```

#### Save

```python
# reddit_dims.reset_index().to_feather(
#     "../data/dims/reddit.feather.zstd", compression="zstd"
# )
default_dims.reset_index().to_feather(
    data_path("dims/reddit.feather.zstd"), compression="zstd"
)
```
