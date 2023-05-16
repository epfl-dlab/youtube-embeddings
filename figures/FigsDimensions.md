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
import pandas as pd

from youtube_topics import data_path
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
reddit_dim = reddit_dim.set_index("index").loc[reddit_embed.index]

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

#### Read channel names, make them unique

```python
chann = pd.read_feather(
    data_path("filtered_channels_default_playlist_flattened.feather.zstd")
)
chann["chanId"] = chann["playlistId"].apply(lambda x: "UC" + x[2:])
chan2 = chann.set_index("chanId")[["snippet.title"]].join(reddit_dim[[]], how="inner")


# deduplicate
test = chan2.reset_index()

test["id"] = test.groupby("snippet.title").cumcount().add(1)

test["chanName"] = test["snippet.title"]
test.loc[test.id != 1, "chanName"] = (
    test[test.id != 1]["snippet.title"]
    + " ("
    + test[test.id != 1]["id"].astype(str)
    + ")"
)

nondup_channame = test.set_index("index")["chanName"]
```

#### Add channel names

```python
cluster_df = (
    dim_topic[["topic"]].rename_axis("name").rename(columns={"topic": "cluster_name"})
)
cluster_df["cluster_id"] = cluster_df.cluster_name.astype("category").cat.codes
cluster_df = cluster_df.join(nondup_channame).set_index("chanName").rename_axis("name")

reddit_dim_nondup = (
    reddit_dim.join(nondup_channame).set_index("chanName").rename_axis("community")
)
```

## Plot percentilized

```python jupyter={"outputs_hidden": true}
many_densities_plot(
    ["age", "gender", "partisan"], cluster_df, reddit_dim_nondup, percentilize=True
);
```

## Plot percentilized singlecol

```python
many_densities_plot(
    ["age", "gender", "partisan"], cluster_df, reddit_dim_nondup, single_column=True
)
plt.savefig(data_path("figures_out/dimensions_singlecol.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

## Plot large with channel names

```python
many_densities_plot(
    ["age", "gender", "partisan"], cluster_df, reddit_dim_nondup, percentilize=False
)
plt.savefig(data_path("figures_out/dimensions_fullcol.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```
