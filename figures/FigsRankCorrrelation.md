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

# Plotting kendall's tau-c rank correlation

In this notebook, we re-read the data obtained from the social dimension rank-correlation to plot them.

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

### Load data

```python
replace_dict = {
    "reddit": "Reddit",
    "content": "Content",
    "recomm": "Recommendation",
    "reddit_avg": "Subreddit",
}

ord_manual = pd.read_csv(data_path("figures_in/dimgen_plot_df_new.csv")).replace(replace_dict)
ord_auto = pd.read_csv(data_path("figures_in/dimgen_plot_df_auto.csv")).replace(replace_dict)
ord_label = pd.read_csv(data_path("figures_in/ord_scores_train_label.csv"))
ord_reddit = pd.read_csv(data_path("figures_in/ordering_train_reddit.csv")).rename(
    columns=replace_dict
)
```

### Plot (vertical)

```python
SCALE = 2
PAGE_WIDTH = 3.324
PAGE_HEIGHT = 9.03
SB_BASELINE_TEXT = "Dimensions from subreddit vectors"

fig, axs = plt.subplots(
    nrows=4,
    figsize=(SCALE * PAGE_WIDTH, SCALE * PAGE_HEIGHT),
    gridspec_kw={"hspace": 0.2},
)
axiter = iter(enumerate(axs))

i, ax = next(axiter)
sns.barplot(
    ord_auto[ord_auto.embed.isin(["Recommendation", "Reddit", "Content"])],
    x="embed",
    y="score",
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_auto.query('embed == "Subreddit"').iloc[0]["score"]
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.95 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title="$\\bf{(a)}$ Kendall's τ : (CNN, Fox News) seed",
    xlabel="",
    ylabel="Kendall's τ",
)

i, ax = next(axiter)
sns.barplot(
    ord_manual[ord_manual.embed.isin(["Recommendation", "Reddit", "Content"])],
    x="embed",
    y="score",
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_manual.query('embed == "Subreddit"').iloc[0]["score"]
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.95 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(title="$\\bf{(b)}$ Kendall's τ : Manual seed", xlabel="", ylabel="Kendall's τ")

i, ax = next(axiter)
sns.barplot(
    ord_label[["Recommendation", "Reddit", "Content"]],
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_label["Subreddit"].mean()
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.95 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title="$\\bf{(c)}$ Kendall's τ  : Binary classifier",
    xlabel="",
    ylabel="Kendall's τ",
)

i, ax = next(axiter)
sns.barplot(
    ord_reddit[["Recommendation", "Reddit", "Content"]],
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_reddit["Subreddit"].mean()
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.95 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title="$\\bf{(d)}$ Kendall's τ  : Dimension regressor",
    xlabel="",
    ylabel="Kendall's τ",
)

for ax in axs:
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", padding=-25, c="w")

set_size(fig, (SCALE * PAGE_WIDTH, SCALE * PAGE_HEIGHT), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_kendall.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

### Same but horizontal

```python
SCALE = 2
PAGE_WIDTH = 1.724
PAGE_HEIGHT = 9.03
SB_BASELINE_TEXT = "Dimensions from subreddit vectors"

fig, axs = plt.subplots(
    ncols=4,
    figsize=(SCALE * PAGE_HEIGHT, SCALE * PAGE_WIDTH),
    gridspec_kw={"hspace": 0.2},
    sharey=True,
)
axiter = iter(enumerate(axs))

i, ax = next(axiter)
sns.barplot(
    ord_auto[ord_auto.embed.isin(["Recommendation", "Reddit", "Content"])],
    x="embed",
    y="score",
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_auto.query('embed == "Subreddit"').iloc[0]["score"]
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.6 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title="$\\bf{(a)}$ Kendall's τ : (CNN, Fox News) seed",
    xlabel="",
    ylabel="Kendall's τ",
)

i, ax = next(axiter)
sns.barplot(
    ord_manual[ord_manual.embed.isin(["Recommendation", "Reddit", "Content"])],
    x="embed",
    y="score",
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_manual.query('embed == "Subreddit"').iloc[0]["score"]
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.6 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(title="$\\bf{(b)}$ Kendall's τ : Manual seed", xlabel="", ylabel="Kendall's τ")

i, ax = next(axiter)
sns.barplot(
    ord_label[["Recommendation", "Reddit", "Content"]],
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_label["Subreddit"].mean()
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.8 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title="$\\bf{(c)}$ Kendall's τ  : Binary classifier",
    xlabel="",
    ylabel="Kendall's τ",
)

i, ax = next(axiter)
sns.barplot(
    ord_reddit[["Recommendation", "Reddit", "Content"]],
    order=["Content", "Recommendation", "Reddit"],
    ax=ax,
)
barheight = ord_reddit["Subreddit"].mean()
ax.axhline(barheight, c="r")
ax.text(-0.48, barheight - barheight / 0.8 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title="$\\bf{(d)}$ Kendall's τ  : Dimension regressor",
    xlabel="",
    ylabel="Kendall's τ",
)

for ax in axs:
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", padding=-15, c="w")

set_size(fig, (SCALE * PAGE_HEIGHT, SCALE * PAGE_WIDTH), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_kendall_wide.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

### MAE and F1

```python
mae = pd.read_csv(data_path("figures_in/mae_train_reddit.csv")).rename(columns=replace_dict)
f1 = pd.read_csv(data_path("figures_in/f1_score_train_label.csv"))
```

```python
fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(mae, ax=ax, order=["Content", "Recommendation", "Reddit"])

ax.set(title="MAE on Dimension regressor")

for cont in ax.containers:
    ax.bar_label(cont, fmt="%.2f", padding=-20, c="w")

set_size(fig, (7, 5), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_mae.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

```python
fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(f1, ax=ax, order=["Content", "Recommendation", "Reddit"])

ax.set(title="F1 on Binary classifier")

for cont in ax.containers:
    ax.bar_label(cont, fmt="%.2f", padding=-30, c="w")

set_size(fig, (7, 5), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_f1.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

### ECDF

```python
df_plot = pd.read_csv(data_path("figures_in/ecdf_plot_df.csv"))
```

```python
fig, axs = plt.subplots(nrows=2, figsize=(7, 10))

sns.ecdfplot(df_plot.query('bias == "left"'), x="partisan", hue="embed", ax=axs[0])
axs[0].set(
    title=r"$\bf{(a)}$ ECDF plot of partisan score for channels labelled $\bf{left}$"
)

sns.ecdfplot(
    df_plot.assign(partisan=1 - df_plot.partisan).query('bias == "right"'),
    x="partisan",
    hue="embed",
    ax=axs[1],
)
axs[1].set(
    title=r"$\bf{(b)}$ ECDF plot of (1-partisan) score for channels labelled $\bf{right}$",
    xlabel="1 - partisan",
)

set_size(fig, (7, 10), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_ecdf.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

### Plot partisan-ness

```python
pness_df = pd.read_csv(data_path("figures_in/pness_labels.csv"))
```

```python
fig, ax = plt.subplots(figsize=(7, 5))

sns.violinplot(pness_df, x="pness", y="bias")
ax.set(
    title="Distribution of partisan-ness for labelled and unlabelled channels",
    xlabel="Partisan-ness",
    ylabel="Label",
)

set_size(fig, (7, 5), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/partisan_ness_labels.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```
