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

# Similarity

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
from youtube_topics.plotting.utils import set_size
from youtube_topics.plotting.similarity import fraction_plot_df, plot_df_similarity, plot_embed_similarity


from youtube_topics import data_path

sns.set_style("whitegrid")
```

## Semantics 


### Load data

```python
full_df_agreement = pd.read_json(
    data_path("figures_in/mturk_similarity_results_final.jsonl"), lines=True
)

with_overall = pd.concat(
    (full_df_agreement, full_df_agreement.assign(embed_data="overall"))
)
```

## Actual plot 

```python
embds = ["overall", "reddit", "content", "recomm"]
plot_dfs = {embed: plot_df_similarity(with_overall, embed) for embed in embds}

fig, ax_dict = plt.subplot_mosaic(
    [
        ["fraction", "overall", "reddit"],
        ["fraction", "overall", "content"],
        ["fraction", "overall", "recomm"],
    ],
    figsize=(20, 7),
    sharey=True,
    gridspec_kw={"hspace": 0.05, "wspace": 0.3},
)

# fraction plot
ax = ax_dict["fraction"]
fraction_plot_df(full_df_agreement, ax)
ax.set_title(
    r"$\bf{(a)}$ Agreement between Workers and Embedding as a function of q",
    fontsize=12,
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [x.capitalize() for x in labels], loc="upper left")


# overall plot
ax = ax_dict["overall"]
plot_embed_similarity(ax, plot_dfs["overall"])
ax.set_title(r"$\bf{(b)}$ Overall Agreement between Workers and Embedding", fontsize=12)
ax.set_xlabel("Minimum number of workers agreeing", fontsize=12)
ax.set_ylabel("Overall Agreement", fontsize=12)


# smaller plots

# reddit
ax = ax_dict["reddit"]
plot_embed_similarity(ax, plot_dfs["reddit"])
ax.set(title=r"$\bf{(c)}$ Agreement between Workers and Embedding per batch")
ax.xaxis.set_visible(False)
ax.set_ylabel(
    "Batch $\\bf{Reddit}$\nAgreement", rotation=0, ha="right", fontsize=12, va="center"
)
ax.set_ylim(top=1)

# content
ax = ax_dict["content"]
plot_embed_similarity(ax, plot_dfs["content"])
ax.set(ylabel="Agreement - Content")
ax.xaxis.set_visible(False)
ax.set_ylabel(
    "Batch $\\bf{Content}$\nAgreement", rotation=0, ha="right", fontsize=12, va="center"
)
ax.set_ylim(top=1)


# recomm
ax = ax_dict["recomm"]
plot_embed_similarity(ax, plot_dfs["recomm"])
ax.set_xlabel("Minimum number of workers agreeing", fontsize=12)
ax.set_ylabel(
    "Batch $\\bf{Recomm}$\nAgreement", rotation=0, ha="right", fontsize=12, va="center"
)
ax.set_ylim(top=1)

set_size(fig, (20, 7), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_similarity.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```
