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
scores_df = pd.read_json(data_path('figures_in/topic_classification.jsonl'), lines=True).replace('Recomm', 'Recommendation')

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
ax.legend(handles, [replace_dict[x] for x in labels], loc="upper left")


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
    hue_order=['Reddit', 'Content', 'Recommendation']
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

### Semantic experiment tables

```python
## FIRST PLOT

order = ['Reddit','Content','Recommendation']
frac_col = 'k'

first_ax_df = (fraction_plot_df(full_df_agreement, ax=None)
               .applymap(lambda x: replace_dict.get(x, x))
               .rename(columns={'embed':'Embedding','frac':frac_col, 'agg':'Value'})
               .pivot(index='Embedding', columns=frac_col, values='Value')
              )

rest_frac = list(set(first_ax_df.T.reset_index().columns) - set([frac_col]))

first_ax_df = first_ax_df.loc[order]
first_ax_df.index = order

print(first_ax_df
      .T.reset_index()
      .style.format('{:.2f}', subset=rest_frac)
      .format(lambda x: f'{x*44000:n}', subset=[frac_col])
      .highlight_max(subset=None, props='textbf:--rwrap', axis=1)
      .hide(axis=0)
      .format_index(lambda x: f'\\textbf{{{x}}}', axis='columns')
      .to_latex(column_format="c|ccc", hrules=True, caption='Agreement between workers and embedding as a function of k', label='table_agreement_k'))

## SECOND PLOT
worker_col = 'Min \#Workers Agreeing'
second_ax_df = (plot_dfs["overall"].applymap(lambda x: replace_dict.get(x, x))
               .rename(columns={'embed':'Embedding','agg_num':worker_col, 'agg':'Value'})
               .pivot(index='Embedding', columns=worker_col, values='Value')
              )


rest_worker = list(set(second_ax_df.T.reset_index().columns) - set([worker_col]))


second_ax_df = second_ax_df.loc[order]
second_ax_df.index = order

print(second_ax_df
      .T.reset_index()
      .style.format('{:.2f}')
      .highlight_max(props='textbf:--rwrap', axis=1, subset=rest_worker)
      .hide(axis=0)
      .format_index(lambda x: f'\\textbf{{{x}}}', axis='columns')
      .to_latex(column_format="c|ccc", hrules=True, caption='Agreement between workers and embedding as a function of minimum \#Workers agreeing', label='table_agreement_minwork'))

## THIRD PLOT
scores_df['avg_score'] = scores_df['scores'].apply(np.mean)
third_ax_df = (scores_df
               .rename(columns={'embed':'Embedding','topic':'Category', 'avg_score':'Value'})
               .pivot(index='Embedding', columns='Category', values='Value')
              )

third_ax_df = third_ax_df.loc[order]
third_ax_df.index = order

print(third_ax_df
      .T.reset_index()
      .style.format('{:.2f}', subset=['Reddit','Content','Recommendation'])
      .highlight_max(props='textbf:--rwrap', axis=1, subset=rest_worker)
      .hide(axis=0)
      .format_index(lambda x: f'\\textbf{{{x}}}', axis='columns')
      .to_latex(column_format="c|ccc", hrules=True, caption='F1 Score per category', label='table_f1_cat'))
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
    ord_reddit.rename(columns={"Subreddit":"Reddit", "Reddit":"Subreddit"})[["Reddit", "Content","Recommendation"]],
    order=["Reddit","Content","Recommendation"],
    ax=ax,
)
# barheight = ord_reddit["Subreddit"].mean()
# ax.axhline(barheight, c="r")
# ax.text(-0.48, barheight - barheight / 0.95 * 0.05, SB_BASELINE_TEXT, ha="left")
ax.set(
    title=r"$\bf{(a)}$ Partisan - Political channel categories",
    xlabel="",
    ylabel="Rank correlation (Kendall's τ)",
)

ax = ax_dict['bt_partisan']
plot2_df = plot_corr_csv(data_path("bradley_terry/bt_partisan_res.csv"), "partisan", default_dims, ax=ax, container_label=False, replace_dict=replace_dict, use_hue=False, order=['Reddit', 'Content', 'Recommendation'])
ax.set(title=r'$\bf{(b)}$ Partisan - "News & Politics" category BT', ylabel='', xlabel='')
# ax.get_legend().remove()

ax = ax_dict['bt_gender']
plot3_df = plot_corr_csv(data_path("bradley_terry/bt_gender_res.csv"), "gender", default_dims, ax=ax, container_label=False, replace_dict=replace_dict, use_hue=False, order=['Reddit', 'Content', 'Recommendation'])
ax.set(title=r'$\bf{(c)}$ Gender - "Howto & Style" category BT', ylabel='', xlabel='')
# ax.get_legend().remove()

ax = ax_dict['bt_age']
plot4_df = plot_corr_csv(data_path("bradley_terry/bt_age_res.csv"), "age", default_dims, ax=ax, container_label=False, replace_dict=replace_dict, use_hue=False, order=['Reddit', 'Content', 'Recommendation'])
ax.set(title=r'$\bf{(d)}$ Age - "Music" category BT', ylabel='', xlabel='')
# ax.get_legend().remove()

for ax in ax_dict.values():
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", padding=-12.5, c="w")
        
set_size(fig, (14, 3.5), eps=1e-3, dpi=300)
plt.savefig(data_path("figures_out/fig_social_dimensions.pdf"), dpi=300, bbox_inches="tight", pad_inches=0)
```

## Dimension tables

```python
plot1_df = ord_reddit.rename(columns={"Subreddit":"Reddit", "Reddit":"Subreddit"})[["Reddit", "Content","Recommendation"]]
    

plot_df = pd.concat([plot1_df.assign(Scoring='Partisan Labels').set_index('Scoring')] +
                    [x.rename(columns={'embed':'Scoring'}).set_index('Scoring').T for x in [plot2_df, plot3_df, plot4_df]])

print(plot_df[['Reddit','Content','Recommendation']]
      .style.format('{:.2f}', subset=None)
      .format_index(lambda x: {'partisan':'BT Partisan', 
                                      'gender':'BT Gender',
                                      'age':'BT Age'
                                     }.get(x,x))
      .highlight_max(subset=None, props='textbf:--rwrap', axis=1)
      .format_index(lambda x: f'\\textbf{{{x}}}', axis='columns')
      .to_latex(hrules=True,
                caption='Social Dimensions experiments table',
                label='dims_table',
                clines="skip-last;data",
                position_float="centering",
                multicol_align="|c|",
               ))
```
