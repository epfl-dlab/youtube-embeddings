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

```python
%load_ext autoreload
%autoreload 2

# isort: off
import sys

sys.path += [".."]
# isort: on

import pandas as pd
import plotly.express as px
from hdbscan import HDBSCAN
from umap import UMAP

from youtube_topics import data_path
```

### Load data

```python
reddit = pd.read_feather(data_path("embeds/reddit.feather.zstd")).set_index("channelId")
recomm = pd.read_feather(data_path("embeds/recomm.feather.zstd")).set_index("channelId")
content = pd.read_feather(data_path("embeds/content.feather.zstd")).set_index(
    "channelId"
)

chan_title_df = pd.read_feather(data_path("channel_title.feather.zstd")).set_index(
    "channelId"
)
```

### Getting UMAP / clustering

```python
def get_umap_df(
    embed_df,
    umap_params=dict(
        n_neighbors=30, min_dist=0.0, n_components=2, random_state=42, n_jobs=20
    ),
    hdbscan_params=dict(min_samples=5, min_cluster_size=50),
):
    # get umap dataframe
    umap_model = UMAP(**umap_params)
    reduced_reddit = umap_model.fit_transform(embed_df)
    umap_reddit = pd.DataFrame(reduced_reddit, index=embed_df.index, columns=["x", "y"])

    # add cluster label
    hdb = HDBSCAN(**hdbscan_params)
    hdb.fit(umap_reddit)
    umap_with_label = umap_reddit.assign(cluster=[str(i) for i in hdb.labels_])

    return umap_with_label


def plot_df(embed, chan_title_df):
    umapdf = get_umap_df(embed)

    return umapdf.join(chan_title_df).reset_index()
```

```python
reddit_plot = plot_df(reddit, chan_title_df)
recomm_plot = plot_df(recomm, chan_title_df)
content_plot = plot_df(content, chan_title_df)
```

### Plotting

```python
fig_reddit = px.scatter(
    reddit_plot, x="x", y="y", color="cluster", hover_data=["channelTitle"]
)
fig_recomm = px.scatter(
    recomm_plot, x="x", y="y", color="cluster", hover_data=["channelTitle"]
)
fig_content = px.scatter(
    content_plot, x="x", y="y", color="cluster", hover_data=["channelTitle"]
)
```

```python
fig_reddit
```

```python
fig_reddit.write_html(data_path("figures_out/reddit_umap.html"))
fig_recomm.write_html(data_path("figures/recomm_umap.html"))
fig_content.write_html(data_path("figures_out/content_umap.html"))
```
