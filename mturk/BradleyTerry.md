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

# Bradley Terry experiment

In this notebook, we prepare the data for the rank correlation experiments using the bradley terry model.

The goal is to make sure that ranks obtained across social dimensions are well correlated with ranks from a bradley terry model.

```python
%load_ext autoreload
%autoreload 2
```

```python
from collections import Counter

# isort: off
import sys

sys.path += [".."]
# isort: on

import pickle

import innertube
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import regex as re
import scipy
import seaborn as sns
from plotly.offline import plot

from youtube_topics import data_path
from youtube_topics.bradley_terry import get_rank, topic_refetch, get_bt_topic, plot_dim_bins, plot_corr_csv, plot_res_scatter
from youtube_topics.mturk import channel_df, mturkify, prep_mturk_batch

sns.set_style("whitegrid")
```

## Plot Reddit dimensions (partisan, partisanness)

Here, we select a way to partition our dataset across the partisan and partisanness dimensions.

```python
default_dims = pd.read_feather(data_path("dims/reddit.feather.zstd")).set_index(
    "channelId"
)
aggdf = pd.read_feather(data_path("per_channel_aggs.feather.zstd"))
topicdf = pd.read_json(data_path("id_to_topic.jsonl.gz"), lines=True)
aggdf["topic_id"] = aggdf["major_cat"].astype(int)
aggdf_topic = aggdf.merge(topicdf, on="topic_id")


## Name df

chan_title = pd.read_feather(data_path('channel_title.feather.zstd')).set_index('channelId')
is_dup = chan_title['channelTitle'].duplicated()
chan_title['channelDedup'] = chan_title['channelTitle']
chan_title.loc[is_dup, 'channelDedup'] = chan_title.loc[is_dup, 'channelDedup'] + chan_title.loc[is_dup].index
chan_title['channelTitle'] = chan_title['channelDedup']
del chan_title['channelDedup']
title_series = chan_title['channelTitle']

fulldim = default_dims.join(title_series).join(
    aggdf_topic.set_index("channelId")[["topic"]]
)
```

```python
px.scatter(fulldim, x="partisan", y="partisan-ness", color="topic", hover_data=['channelTitle'])
```

## Sampling specific categories with specific dimensions

We select youtube channels from categories which are heavily linked with a particular dimension:

- Howto & style - Gender
- News & Politics - Partisan
- Music - Age

We then partition our data in bins based on the dimension and dimension-ness score (within 0,5 std, 1.5 std, etc), and sample hierarchically from those bins to create our bradley terry experiment.

While those experiments are technically seeded, you might get different samplings at different times if channels are deleted, deleting all their videos, etc..

```python
# bin borders
xs = [-5, -1.5, -0.5, 0.5, 1.5, 5]
ys = [-5, 0, 5]
```

### Howto & style - Gender

```python
dim = "gender"
howto_gender = topic_refetch(fulldim, "Howto & Style", xs, ys, dim)
howto_gender_bt = get_bt_topic(howto_gender, 6, 20, seed=5).assign(dim=dim)
howto_gender_bt.to_csv(data_path("bt/howto_gender_bt.csv"), index=False)
plot_dim_bins(howto_gender, xs, ys, xcol="topicdim", ycol="topicness")
```

### News & Politics - Partisan

```python
dim = "partisan"
news_partisan = topic_refetch(fulldim, "News & Politics", xs, ys, dim)
news_partisan_bt = get_bt_topic(news_partisan, 6, 20, seed=5).assign(dim=dim)
news_partisan_bt.to_csv(data_path("bt/news_partisan_bt.csv"), index=False)
plot_dim_bins(news_partisan, xs, ys, xcol="topicdim", ycol="topicness")
```

### Music - Age

```python
dim = "age"
music_age = topic_refetch(fulldim, "Music", xs, ys, dim)
music_age_bt = get_bt_topic(music_age, 6, 20, seed=5).assign(dim=dim)
music_age_bt.to_csv(data_path("bt/music_age_bt.csv"), index=False)
plot_dim_bins(music_age, xs, ys, xcol="topicdim", ycol="topicness")
```

## Read results, get Bradley-Terry (/Plackett-Luce) rankings

```python
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

plot_corr_csv(data_path("bt/news_partisan_res.csv"), "partisan", default_dims, ax=axs[0])
axs[0].set(title="Partisan - News category correlation")

plot_corr_csv(data_path("bt/howto_gender_res.csv"), "gender", default_dims, ax=axs[1], reverse_dim=False)
axs[1].set(title="Gender - Howto category correlation")

plot_corr_csv(data_path("bt/music_age_res.csv"), "age", default_dims, ax=axs[2])
axs[2].set(title="Age - Music category correlation")

plt.suptitle("Correlation between Bradley Terry ranks & Regression training ranks")
```

```python
plot_res_scatter(
    data_path("bt/news_partisan_res.csv"),
    data_path("bt/news_partisan_scatter.html"),
    "partisan",
    title_series
)
plot_res_scatter(
    data_path("bt/howto_gender_res.csv"),
    data_path("bt/howto_gender_scatter.html"),
    "gender",
    title_series,
    reverse_dim=False
)
plot_res_scatter(
    data_path("bt/music_age_res.csv"),
    data_path("bt/music_age_scatter.html"),
    "age",
    title_series
)
```
