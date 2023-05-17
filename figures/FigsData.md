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

# Plotting data distributions

Here, we plot the most relevant data distributions (posts per author, mentions per channel, subreddits on which channels are mentionned, in and out degrees of nodes).

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
import seaborn as sns
from youtube_topics.plotting.utils import set_size

from youtube_topics import data_path

sns.set_style("whitegrid")
```

### Load DataFrames

```python
# files created in RedditEmbed.ipynb
authors = pd.read_feather(data_path("figures_in/author_mentions.feather.zstd"))
channels_mentions = pd.read_feather(data_path("figures_in/channel_mentions.feather.zstd"))
channel_subreddits = pd.read_feather(data_path("figures_in/channel_subreddits.feather.zstd"))
in_deg = pd.read_feather(data_path("figures_in/in_degree_distribution.feather.zstd"))
out_deg = pd.read_feather(data_path("figures_in/out_degree_distribution.feather.zstd"))
author_s = authors.set_index("author")["count"].loc[lambda x: x != 0]

channel_mention_s = channels_mentions.set_index("channelId")["mentions"].loc[
    lambda x: x != 0
]
channel_subreddit_s = channel_subreddits.set_index("channelId")["subreddits"].loc[
    lambda x: x != 0
]

in_deg_s = in_deg.set_index("node")["in_degree"]
out_deg_s = out_deg.set_index("node")["out_degree"]
```

### Plot

```python
SCALE = 2
PAGE_WIDTH = 3.324
PAGE_HEIGHT = 9.03

fig, axs = plt.subplots(
    nrows=5,
    figsize=(SCALE * PAGE_WIDTH, SCALE * PAGE_HEIGHT),
    gridspec_kw={"hspace": 0.4},
)
axiter = iter(enumerate(axs))

# authors
i, ax = next(axiter)
sns.histplot(author_s, bins=40, log_scale=(True, True), ax=ax, color=f"C{i}")
ax.set(
    title=r"$\bf{(a)}$ Distribution of number of video posts per Reddit author",
    xlabel="Number of posts per author",
)

# mentions per chan
i, ax = next(axiter)
sns.histplot(channel_mention_s, bins=40, log_scale=(True, True), ax=ax, color=f"C{i}")
ax.set(title=r"$\bf{(b)}$ Number of mentions per channel", xlabel="Number of mentions")

# subreddits per chan
i, ax = next(axiter)
sns.histplot(channel_subreddit_s, bins=40, log_scale=(True, True), ax=ax, color=f"C{i}")
ax.set(
    title=r"$\bf{(c)}$ Number of subreddits on which channels are mentioned",
    xlabel="Number of subreddits",
)

# in degree
i, ax = next(axiter)
sns.histplot(in_deg_s + 1, bins=40, log_scale=(True, True), ax=ax, color=f"C{i}")
ax.set(
    title=r"$\bf{(d)}$ Distribution of the in-degree of nodes", xlabel="In-degree + 1"
)

# out degree
i, ax = next(axiter)
sns.histplot(out_deg_s + 1, bins=40, log_scale=(True, True), ax=ax, color=f"C{i}")
ax.set(
    title=r"$\bf{(e)}$ Distribution of the out-degree of nodes", xlabel="Out-degree + 1"
)

set_size(fig, (SCALE * PAGE_WIDTH, SCALE * PAGE_HEIGHT), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_data.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```
