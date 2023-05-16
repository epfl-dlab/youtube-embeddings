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

chan_name_df = pd.read_feather(
    data_path("old_files/filtered_channels_videos_flattened.feather.zstd")
)
name_df = (
    chan_name_df[["snippet.channelId", "snippet.channelTitle"]]
    .drop_duplicates()
    .rename(columns={"snippet.channelId": "channelId", "snippet.channelTitle": "title"})
)
del chan_name_df
title_series = name_df.set_index("channelId")["title"]

fulldim = default_dims.join(title_series).join(
    aggdf_topic.set_index("channelId")[["topic"]]
)
```

```python
px.scatter(fulldim, x="partisan", y="partisan-ness", color="topic")
```

## Sampling specific categories with specific dimensions

We select youtube channels from categories which are heavily linked with a particular dimension:

- Howto & style - Gender
- News & Politics - Partisan
- Music - Age

We then partition our data in bins based on the dimension and dimension-ness score (within 0,5 std, 1.5 std, etc), and sample hierarchically from those bins to create our bradley terry experiment.

```python
xs = [-5, -1.5, -0.5, 0.5, 1.5, 5]
ys = [-5, 0, 5]

```

### Howto & style - Gender

```python
dim = "gender"
howto_gender = topic_refetch(fulldim, "Howto & Style", xs, ys, dim)
howto_gender_bt = get_bt_topic(howto_gender, 6, 20, seed=5).assign(dim=dim)
howto_gender_bt.to_csv(data_path("howto_gender_bt.csv"), index=False)
plot_dim_bins(howto_gender, xs, ys, xcol="topicdim", ycol="topicness")
```

### News & Politics - Partisan

```python
dim = "partisan"
news_partisan = topic_refetch(fulldim, "News & Politics", xs, ys, dim)
news_partisan_bt = get_bt_topic(news_partisan, 6, 20, seed=5).assign(dim=dim)
news_partisan_bt.to_csv(data_path("news_partisan_bt.csv"), index=False)
plot_dim_bins(news_partisan, xs, ys, xcol="topicdim", ycol="topicness")
```

### Music - Age

```python
dim = "age"
music_age = topic_refetch(fulldim, "Music", xs, ys, dim)
music_age_bt = get_bt_topic(music_age, 6, 20, seed=5).assign(dim=dim)
music_age_bt.to_csv(data_path("music_age_bt.csv"), index=False)
plot_dim_bins(music_age, xs, ys, xcol="topicdim", ycol="topicness")
```

```python
path = data_path("bt_results/news_partisan_resnew.csv")
dimension = "partisan"
df = pd.read_csv(path)
df_rank = get_rank(df, dimension)
df_rank.to_csv(data_path("rankdfs/news_partisan.csv"))

path = data_path("bt_results/howto_gender_resnew.csv")
dimension = "gender"
df = pd.read_csv(path)
df_rank = get_rank(df, dimension)
df_rank.to_csv(data_path("rankdfs/howto_gender.csv"))

path = data_path("bt_results/music_age_resnew.csv")
dimension = "age"
df = pd.read_csv(path)
df_rank = get_rank(df, dimension)
df_rank.to_csv(data_path("rankdfs/music_age.csv"))
```

```python
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

plot_corr_csv(data_path("bt_results/news_partisan_resnew.csv"), "partisan", default_dims, ax=axs[0])
axs[0].set(title="Partisan, news category corr")

plot_corr_csv(data_path("bt_results/howto_gender_resnew.csv"), "gender", default_dims, ax=axs[1])
axs[1].set(title="Howto, gender category corr")

plot_corr_csv(data_path("bt_results/music_age_resnew.csv"), "age", default_dims, ax=axs[2])
axs[2].set(title="Music, age category corr")

plt.suptitle("Correlation between Bradley Terry ranks & Regression training ranks")
```

```python
plot_res_scatter(
    data_path("bt_results/news_partisan_resnew.csv"),
    data_path("news_partisan_scatter.html"),
    "partisan",
    title_series
)
plot_res_scatter(
    data_path("bt_results/howto_gender_resnew.csv"),
    data_path("howto_gender_scatter.html"),
    "gender",
    title_series
)
plot_res_scatter(
    data_path("bt_results/music_age_resnew.csv"),
    data_path("music_age_scatter.html"),
    "age",
    title_series
)
```
