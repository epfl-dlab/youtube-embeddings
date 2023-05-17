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

```python
%load_ext autoreload
%autoreload 2
```

```python
import json
import logging
import time
from collections import defaultdict
from functools import reduce
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import sys

sys.path += ['..']

from youtube_topics import data_path
plt.style.use("ggplot")
```

## Load embeddings, YouTube categories

```python
embeds_dic = dict(
    reddit=pd.read_feather(data_path('embeds/reddit.feather.zstd')).set_index('channelId'),
    content=pd.read_feather(data_path('embeds/content.feather.zstd')).set_index('channelId'),
    recomm=pd.read_feather(data_path('embeds/recomm.feather.zstd')).set_index('channelId'))

### Load topics from fetched api v3 data
aggdf = pd.read_feather(data_path("per_channel_aggs.feather.zstd"))
topicdf = pd.read_json(data_path("id_to_topic.jsonl.gz"), lines=True)
aggdf["topic_id"] = aggdf["major_cat"].astype(int)
aggdf_topic = aggdf.merge(topicdf, on="topic_id")

### Keep intersecting channels
inters = sorted(
    reduce(
        set.intersection,
        [set(aggdf_topic.channelId)] + [set(df.index) for df in embeds_dic.values()],
    )
)
aggdf_topic = aggdf_topic.set_index("channelId").loc[inters].reset_index()

### Single dataframe for all embeds
embeds_df = reduce(
    pd.DataFrame.join,
    [df.assign(**{name: list(df.values)})[[name]] for name, df in embeds_dic.items()],
)
embeds_df = embeds_df.dropna()
```

### Bootstrap sample

```python
cats = "Gaming", "Music", "Sports"
chantopic = aggdf_topic[["channelId", "topic"]]

N_SAMPLE = 100
NUM_ITERS = 100
datasets = {}

for cat in cats:

    pos_ids = []
    neg_ids = []

    for _ in tqdm(range(NUM_ITERS)):

        pos_df = chantopic.query("topic == @cat").sample(N_SAMPLE, replace=False)
        neg_df = chantopic.query("topic != @cat").sample(N_SAMPLE, replace=False)

        pos_ids.append(pos_df["channelId"].values)
        neg_ids.append(neg_df["channelId"].values)

    datasets[cat] = list(zip(pos_ids, neg_ids))
```

### Run bootstrap preds

```python
scores = defaultdict(list)

# gaming, sport, etc
for cat in datasets:

    cat_df = datasets[cat]

    # get positive, negative samples
    for it, (pos, neg) in tqdm(enumerate(cat_df), total=len(cat_df)):

        # df with neg and pos label
        iterdf = pd.concat(
            (embeds_df.loc[pos].assign(label=1), embeds_df.loc[neg].assign(label=0))
        )

        # train test split (stratify, 70/30), get y
        train_indices, test_indices = train_test_split(
            np.arange(iterdf.shape[0]), stratify=iterdf.label, random_state=it
        )
        y = iterdf["label"]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        cols = [col for col in iterdf.columns if "label" not in col]

        # over all embeddings
        for col in cols:

            # get X
            X = np.vstack(iterdf[col])
            X_train, X_test = X[train_indices], X[test_indices]

            # train
            forest = RandomForestClassifier(random_state=it, n_estimators=100).fit(
                X_train, y_train
            )

            # predict f1
            preds = forest.predict(X_test)
            scores[(cat, col)].append(f1_score(y_test, preds))
```

### Plot preds

```python
scores_df = pd.concat(
    (
        pd.DataFrame(scores.keys(), columns=["topic", "embed"]),
        pd.Series(scores.values()).to_frame("scores"),
    ),
    axis=1,
)
scores_df = scores_df.assign(embed=scores_df.embed.apply(lambda s: s.capitalize()))
```

```python
scores_df.to_json(data_path('figures_in/topic_classification.jsonl'), lines=True, orient='records')
```

```python
# scores_df_comp = pd.read_feather("topic_classification_results.feather.zstd")
```

### Topic class recomm embed

```python
import seaborn as sns
sns.set_style('whitegrid')
```

```python
fig, ax = plt.subplots(figsize=(7, 5))

ordering = (scores_df
            .assign(scoremean=scores_df.scores.apply(np.mean))
            .groupby('embed')['scoremean'].mean().sort_values()
            .index)

sns.barplot(
    scores_df.explode("scores"),
    x="topic",
    hue="embed",
    y="scores",
    ax=ax,
    palette='tab10',
    hue_order=ordering
)

for cont in ax.containers:
    ax.bar_label(cont, padding=15, fmt="%.2f")

ax.set_ylim(ymin=0.7)
ax.set(title='F1-score on topic classification')
ax.legend(loc='lower center')

plt.tight_layout()
#plt.savefig('topic_classification_reddit.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
```

```python
fig, ax = plt.subplots(figsize=(15, 5))

sns.barplot(
    scores_df.explode("scores"),
    x="topic",
    hue="embed",
    y="scores",
    ax=ax,
    palette="tab10",
)

for cont in ax.containers:
    ax.bar_label(cont, padding=5, fmt="%.3f")

ax.set_ylim(ymin=0.7)

ax.legend(loc="lower right")

plt.tight_layout()
```

### Topic class all embeds

```python
fig, ax = plt.subplots(figsize=(15, 5))

sns.barplot(
    scores_df_comp.explode("scores"),
    x="topic",
    hue="embed",
    y="scores",
    ax=ax,
    palette="tab10",
)

for cont in ax.containers:
    ax.bar_label(cont, padding=5, fmt="%.3f")

ax.set_ylim(ymin=0.7)

ax.legend(loc="lower right")

plt.tight_layout()
```
