---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Experiments for selecting comments vs posts

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd

from contextlib import redirect_stderr, redirect_stdout

from youtube_topics import data_path
fprop = fm.FontProperties(fname=data_path('NotoSansCJKkr-Regular.otf'))


from youtube_topics.plotting.social_dim_reddit import many_densities_plot
from youtube_topics.reddit_averaging import *
```

## Read channel, subreddit dataframe

```python
classes = pd.read_csv(data_path("polarized_manoel_mbc_nondup.csv"))
ordering = [
    "extremeleft",
    "left",
    "centerleft",
    "center",
    "centerright",
    "right",
    "extremeright",
]
```

#### Read youtube channel categories

```python
# og
reddit_embed = pd.read_feather(data_path("embeds/reddit.feather.zstd")).set_index(
    "channelId"
)

# get dimension
reddit_dim = pd.read_feather(data_path("dims/reddit.feather.zstd"))
reddit_dim = reddit_dim.set_index("channelId").loc[reddit_embed.index]

# topic dataframe
aggdf = pd.read_feather(data_path("per_channel_aggs.feather.zstd"))
topicdf = pd.read_json(data_path("id_to_topic.jsonl.gz"), lines=True)
aggdf["topic_id"] = aggdf["major_cat"].astype(int)
aggdf_topic = aggdf.merge(topicdf, on="topic_id")

# normalize dims and join with topics
normalized_dim = (reddit_dim - reddit_dim.mean()) / reddit_dim.std()
dim_topic = normalized_dim.join(
    aggdf_topic.set_index("channelId")[["topic"]], how="inner"
)

# colors
topics = dim_topic.topic.value_counts().index
dimensions = ["age", "gender", "partisan"]
cmap_dict = {"partisan": "coolwarm", "gender": "PuOr", "age": "PiYG"}
```

```python
#### Read channel names, make them unique

chan_title = pd.read_feather(data_path('channel_title.feather.zstd')).set_index('channelId')
is_dup = chan_title['channelTitle'].duplicated()
chan_title['channelDedup'] = chan_title['channelTitle']
chan_title.loc[is_dup, 'channelDedup'] = (chan_title.loc[is_dup, 'channelDedup'] +
                                          pd.Series(chan_title.loc[is_dup].index, index=chan_title.loc[is_dup].index).apply(lambda s: f" ({s})"))

chan_title['channelTitle'] = chan_title['channelDedup']
del chan_title['channelDedup']
```

#### Add channel names

```python
cluster_df = (
    dim_topic[["topic"]].rename_axis("name").rename(columns={"topic": "cluster_name"})
)
cluster_df["cluster_id"] = cluster_df.cluster_name.astype("category").cat.codes
cluster_df = cluster_df.join(chan_title).set_index("channelTitle").rename_axis("name")

reddit_dim_nondup = (
    reddit_dim.join(chan_title).set_index("channelTitle").rename_axis("community")
)
```

## Plot percentilized

```python tags=[]
with redirect_stderr(None):
    many_densities_plot(
            ["age", "gender", "partisan"], cluster_df, reddit_dim_nondup, percentilize=True
    );
```

## Plot percentilized singlecol

```python
with redirect_stderr(None):
    many_densities_plot(
        ["age", "gender", "partisan"], cluster_df, reddit_dim_nondup, single_column=True
    )
plt.savefig(data_path("figures_out/dimensions_singlecol.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

## Plot large with channel names

```python
fprop = fm.FontProperties(fname=data_path('NotoSansCJKkr-Regular.otf'))
with redirect_stderr(None), redirect_stdout(None):
    many_densities_plot(
        ["age", "gender", "partisan"], cluster_df, reddit_dim_nondup, percentilize=False, fontproperties=fprop
    )
plt.savefig(data_path("figures_out/dimensions_fullcol.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

```python

```
