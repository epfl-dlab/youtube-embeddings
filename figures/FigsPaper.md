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

# Plotting Mturk Similarity Experiment

Raw results are filtered and processed in the MturkResults notebook.

Here, we load the processed (and public data) to plot it.

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from youtube_topics.plotting.utils import set_size
from youtube_topics.plotting.similarity import fraction_plot_df, plot_df_similarity, plot_embed_similarity
from youtube_topics.bradley_terry import (get_rank,
                                          topic_refetch,
                                          get_bt_topic,
                                          plot_dim_bins, plot_corr_csv, plot_res_scatter, plot_reg_correlation,
                                          dim_corrs)


from youtube_topics import data_path

sns.set_style("whitegrid")
```

### Load data

```python
full_df_agreement = pd.read_json(
    data_path("figures_in/mturk_similarity_results_final.jsonl"), lines=True
)

with_overall = pd.concat(
    (full_df_agreement, full_df_agreement.assign(embed_data="overall"))
)

# topic classification scores
scores_df = pd.read_json(data_path('figures_in/topic_classification.jsonl'), lines=True)

# rank correlation reddit
replace_dict = {
    "reddit": "Reddit",
    "content": "Content",
    "recomm": "Recommendation",
    "reddit_avg": "Subreddit",
}

ord_reddit = pd.read_csv(data_path("figures_in/ordering_train_reddit.csv")).rename(
    columns=replace_dict
)
default_dims = pd.read_feather(data_path("dims/reddit.feather.zstd")).set_index(
    "channelId"
)
```

## Semantic plot

```python
embds = ["overall", "reddit", "content", "recomm"]
plot_dfs = {embed: plot_df_similarity(with_overall, embed) for embed in embds}

fig, ax_dict = plt.subplot_mosaic(
    [
        ["fraction", "overall", "topic"]
    ],
    figsize=(14, 3.5),
    sharey=True,
    gridspec_kw={"hspace": 0.05, "wspace": 0.1},
)

FONTSIZE = 8
mpl.rcParams.update({'font.size': FONTSIZE})

# fraction plot
ax = ax_dict["fraction"]
fraction_plot_df(full_df_agreement, ax, fontsize=FONTSIZE)
ax.set_title(
    r"$\bf{(a)}$ Agreement between Workers and Embedding as a function of q",
    fontsize=FONTSIZE,
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [x.capitalize() for x in labels], loc="upper left")


# overall plot
ax = ax_dict["overall"]
plot_embed_similarity(ax, plot_dfs["overall"])
ax.set_title(r"$\bf{(b)}$ Overall Agreement between Workers and Embedding", fontsize=FONTSIZE)
ax.set_xlabel("Minimum number of workers agreeing", fontsize=FONTSIZE)
ax.set_ylabel("Overall Agreement", fontsize=FONTSIZE)

# topic classification
ax = ax_dict['topic']

sns.barplot(
    scores_df.explode("scores"),
    x="topic",
    hue="embed",
    y="scores",
    ax=ax,
    palette='tab10',
    hue_order=['Reddit', 'Content', 'Recomm']
)

for cont in ax.containers:
    ax.bar_label(cont, padding=3, fmt="%.2f")

ax.set_xlabel('YouTube category', fontsize=FONTSIZE)
ax.set_ylabel('F1 score', fontsize=FONTSIZE)
ax.set_title(r'$\bf{(c)}$ Topic Classification F1-score', fontsize=FONTSIZE)
ax.get_legend().remove()

set_size(fig, (14, 3.5), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_semantic.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

<!-- #region tags=[] -->
## Dimensions plot
<!-- #endregion -->

```python
fig, ax_dict = plt.subplot_mosaic(
    [
        ["rankcorrlabel", "bt_partisan","bt_gender",  "bt_age"]
    ],
    figsize=(14, 3.5),
    sharey=True,
    gridspec_kw={"hspace": 0.05, "wspace": 0.1},
)
SB_BASELINE_TEXT = "Dimensions from subreddit vectors"


ax = ax_dict['rankcorrlabel']
sns.barplot(
    ord_reddit[["Reddit", "Content","Recommendation"]],
    order=["Reddit","Content","Recommendation"],
    ax=ax,
)
barheight = ord_reddit["Subreddit"].mean()
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.95 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title=r"$\bf{(a)}$ Partisan - Political channel categories",
    xlabel="",
    ylabel="Rank correlation (Kendall's τ)",
)

ax = ax_dict['bt_partisan']
plot_corr_csv(data_path("bt/news_partisan_res.csv"), 'partisan', default_dims, ax=ax, legend=False, container_label=False, use_hue=False, order=["Reddit","Content","Recommendation"], y_label=False, x_label=False, replace_dict=replace_dict)
ax.set(title=r'$\bf{(b)}$ Partisan - "News & Politics" category BT')

ax = ax_dict['bt_gender']
plot_corr_csv(data_path("bt/howto_gender_res.csv"), "gender", default_dims, ax=ax, reverse_dim=False, legend=False, container_label=False, use_hue=False, order=["Reddit","Content","Recommendation"], y_label=False, x_label=False, replace_dict=replace_dict)
ax.set(title=r'$\bf{(c)}$ Gender - "Howto & Style" category BT')

ax = ax_dict['bt_age']
plot_corr_csv(data_path("bt/music_age_res.csv"), "age", default_dims, ax=ax, legend=False, container_label=False, use_hue=False, order=["Reddit","Content","Recommendation"], y_label=False, x_label=False,replace_dict=replace_dict)
ax.set(title=r'$\bf{(d)}$ Age - "Music" category BT')

for ax in ax_dict.values():
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", padding=-12.5, c="w")
        
set_size(fig, (14, 3.5), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_social_dimensions.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```
