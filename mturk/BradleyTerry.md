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

import json
import logging
import pickle
import time
from datetime import datetime

import boto3
import innertube
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import regex as re
import scipy
import seaborn as sns
from plotly.offline import plot
from tqdm.auto import tqdm

from youtube_topics import data_path
from youtube_topics.bradley_terry import (BTMTurkHelper, get_bt_topic,
                                          get_rank, plot_corr_csv,
                                          plot_dim_bins, plot_res_scatter,
                                          topic_refetch)
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

chan_title = pd.read_feather(data_path("channel_title.feather.zstd")).set_index(
    "channelId"
)
is_dup = chan_title["channelTitle"].duplicated()
chan_title["channelDedup"] = chan_title["channelTitle"]
chan_title.loc[is_dup, "channelDedup"] = (
    chan_title.loc[is_dup, "channelDedup"] + chan_title.loc[is_dup].index
)
chan_title["channelTitle"] = chan_title["channelDedup"]
del chan_title["channelDedup"]
title_series = chan_title["channelTitle"]

fulldim = default_dims.join(title_series).join(
    aggdf_topic.set_index("channelId")[["topic"]]
)
```

```python
px.scatter(
    fulldim, x="partisan", y="partisan-ness", color="topic", hover_data=["channelTitle"]
)
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
xs = [-5, -1.25, -0.5, 0.5, 1.25, 5]
ys = [-5, 0, 5]
```

### Howto & style - Gender

```python
dim = "gender"
howto_gender = topic_refetch(fulldim, "Howto & Style", xs, ys, dim, mean_method="mean")
howto_gender_bt = get_bt_topic(
    howto_gender,
    n_per_label=10,
    pair_per_channel=20,
    seed=300,
    batch_size=20,
    oversample=1.5,
).assign(dim=dim)
howto_gender_bt.to_csv(data_path("bt/howto_gender_bt_large.csv"), index=False)
plot_dim_bins(howto_gender, xs, ys, xcol="topicdim", ycol="topicness")
```

### News & Politics - Partisan

```python
dim = "partisan"
news_partisan = topic_refetch(
    fulldim, "News & Politics", xs, ys, dim, mean_method="mean"
)
news_partisan_bt = get_bt_topic(
    news_partisan,
    n_per_label=10,
    pair_per_channel=20,
    seed=300,
    batch_size=20,
    oversample=1.5,
).assign(dim=dim)
news_partisan_bt.to_csv(data_path("bt/news_partisan_bt_large.csv"), index=False)
plot_dim_bins(news_partisan, xs, ys, xcol="topicdim", ycol="topicness")
```

### Music - Age

```python
dim = "age"
music_age = topic_refetch(fulldim, "Music", xs, ys, dim, mean_method="mean")
music_age_bt = get_bt_topic(
    music_age,
    n_per_label=10,
    pair_per_channel=20,
    seed=300,
    batch_size=20,
    oversample=1.5,
).assign(dim=dim)
music_age_bt.to_csv(data_path("bt/music_age_bt_large.csv"), index=False)
plot_dim_bins(music_age, xs, ys, xcol="topicdim", ycol="topicness")
```
