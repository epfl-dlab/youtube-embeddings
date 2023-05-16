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

# Creating recommendation embedding from generated graph

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

from glob import glob

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from fastnode2vec import Graph, Node2Vec

from youtube_topics import data_path
```

### Read recommendations 

We read the recommendations we obtained from the innertube api

```python
paths = glob(data_path("innertube_recomms/id_*_df.feather.zstd"))

tmp_df = pd.concat(pd.read_feather(path) for path in paths)
tmp_df["authors"] = tmp_df["result"].apply(lambda l: [x["channelId"] for x in l])
df = tmp_df.rename(columns={"filtered_vid_id": "videoId"})[["videoId", "authors"]]

del tmp_df
```

### Map original video ids to channel ids

We only have video ids to recommended channel id, need to use our dataset to have a final dataframe with (channel, recommended channel).

```python
# df of channels from which we fetch
chan_df = pl.read_ipc(data_path("channel_vid_id.feather.zstd"))
chan_df = chan_df.with_columns(
    chan_df["playlistId"].apply(lambda x: "UC" + x[2:]).alias("channelId")
)

tmp_df = chan_df[["channelId", "result"]].explode("result")
concatdf = pl.concat(
    (tmp_df[["channelId"]], tmp_df["result"].struct.to_frame()), how="horizontal"
)

# concatdf.write_ipc('exploded_channel_vid_df.feather.zstd', compression="zstd")
```

### Use this map to get back per channel ids, thus creating our edge df for creating our graph

```python
vid_df = pd.read_feather(data_path("exploded_channel_vid_df.feather.zstd"))
vid_df = vid_df.drop_duplicates(subset=["videoId"])

# get back channelId
merged = vid_df.merge(df, on="videoId", how="inner")

# explode, one row per channel / recomm pair
exploded = merged[["channelId", "authors"]].explode("authors")

# group by original channel to recommended channel, get weight
weight_df = (
    exploded.groupby(["channelId", "authors"]).size().to_frame("weight").reset_index()
)

# remove self edges
weight_df = weight_df[weight_df["channelId"] != weight_df["authors"]]
weight_df = weight_df.reset_index(drop=True)

# weight_df = pd.read_feather(data_path("graph_edgelist_removed_selfedges.feather.zstd"))
```

## Networkx exploration

```python
G = nx.from_pandas_edgelist(weight_df, "channelId", "authors", create_using=nx.DiGraph)
```

```python
in_degrees = G.in_degree(G.nodes)
out_degres = G.out_degree(G.nodes)

out_deg_s = pd.Series(x[1] for x in out_degres)
in_deg_s = pd.Series(x[1] for x in in_degrees)

in_deg_s.to_frame("in_degree").rename_axis("node").reset_index().to_feather(
    data_path("figures_in/in_degree_distribution.feather.zstd"), compression="zstd"
)
out_deg_s.to_frame("out_degree").rename_axis("node").reset_index().to_feather(
    data_path("figures_in/out_degree_distribution.feather.zstd"), compression="zstd"
)
```

## Node2Vec


#### Training as an undirected graph

```python
undirected_graph = Graph(weight_df.values, directed=False, weighted=True)
n2v_undir = Node2Vec(
    undirected_graph, dim=64, walk_length=40, context=20, p=1.0, q=1.0, workers=8
)

n2v_undir.train(epochs=100)
```

#### Getting back vectors from node2vec, saving them as our recommendation embedding

```python
# df of channels we filter
channame = pd.read_feather(data_path("filtered_channels.feather.zstd"))
n2v = n2v_undir

# get vectors
vecdf = pd.DataFrame(n2v.wv.key_to_index.keys(), columns=["id"])
vecdf["vect"] = list(n2v.wv.vectors)

# filter original channels we were trying to fetch
merged = vecdf.merge(channame[["id", "title"]], on=["id"], how="inner")

# by channel title
merged_exp = (
    pd.DataFrame(np.stack(merged.vect))
    .assign(title=merged.title.tolist())
    .set_index("title")
)

# by channel id
merged_by_id = (
    pd.DataFrame(np.stack(merged.vect))
    .assign(title=merged.id.tolist())
    .set_index("title")
)

# save dataframe with channel id, embeddings
merged_by_id.rename_axis("channelId").reset_index().rename(
    columns=lambda i: str(i)
).to_feather(data_path("recomm64_undirected_q2.feather.zstd"), compression="zstd")
```
