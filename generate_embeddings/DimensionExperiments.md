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

# Creating dimensions from each embedding

In this notebook, we perform the various experiments related to creating and comparing social dimension scores for our embeddings.

As a metric, we use the [Stuart-Kendall $\tau_c$ rank correlation coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Tau-c), between labels of political youtube channels obtained from https://mediabiasfactcheck.com/, and the social dimension scores we obtain.

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

import logging
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

from youtube_topics import data_path
from youtube_topics.scoring import kendalltau_score, kendalltau_scorer

sns.set_style("whitegrid")
logging.getLogger().setLevel(logging.INFO)
```

### Read embeds

```python
reddit_embed = pd.read_feather(data_path("embeds/reddit.feather.zstd")).set_index(
    "channelId"
)
recomm_embed = pd.read_feather(data_path("embeds/recomm.feather.zstd")).set_index(
    "channelId"
)
content_embed = pd.read_feather(data_path("embeds/content.feather.zstd")).set_index(
    "channelId"
)
```

### Read the original Reddit social dimensions

Those YT channel dimensions are the ones we created using the subreddit vectors from Waller, Anderson on our Reddit embedding.

```python
# get dimension
reddit_dim_normal = pd.read_feather(data_path("dims/reddit.feather.zstd"))
reddit_dim_normal = reddit_dim_normal.set_index("channelId").loc[reddit_embed.index]
reddit_dim = reddit_dim_normal
```

#### Create DataFrames from DimensionGenerator seed pairs

```python
def dimdf_from_pairdf(pairdf, normed_embed):
    # get vec
    vec = (
        normed_embed.loc[pairdf["right"]].values
        - normed_embed.loc[pairdf["left"]].values
    ).mean(axis=0)
    vecness = (
        normed_embed.loc[pairdf["right"]].values
        + normed_embed.loc[pairdf["left"]].values
    ).mean(axis=0)
    dim = normed_embed @ (vec / np.linalg.norm(vec))
    ness = normed_embed @ (vecness / np.linalg.norm(vecness))

    # return dim df
    dimdf = (
        dim.sort_values(ascending=True)
        .to_frame("partisan")
        .join(ness.to_frame("partisan-ness"))
    )
    return dimdf
```

#### Utils to get channel Id from channel title

```python
chan_df = pd.read_feather(data_path("channel_title.feather.zstd"))
title_to_id = chan_df.set_index("channelTitle")["channelId"].to_dict()


def get_id(chan_name):
    import difflib

    matches = difflib.get_close_matches(chan_name, title_to_id, n=1)

    if len(matches) == 0:
        raise ValueError(f"No match found for {chan_name}")

    chan = matches[0]

    if chan != chan_name:
        logging.warning(f"No {chan_name} found, reverted to {chan}")

    return title_to_id[chan]
```

#### Utils to compute ordering score

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
joined_class = (
    classes.set_index("channelId").join(reddit_embed[[]], how="inner").reset_index()
)
tau_scorer = kendalltau_scorer(joined_class, ordering, id_col="channelId")
```

## Part 1: Using DimenGenerator

Create dimensions using DimensionGenerator from https://github.com/CSSLab/social-dimensions (slighty modified algo to use sklearn NearestNeighbors so we don't compute unnecessary distances), and youtube channel seeds.

We use the seeds **Fox News**, **CNN** for computing the partisan dimension.

```python
%%time

content_norm = content_embed.divide(
    np.linalg.norm(content_embed.values, axis=1), axis="rows"
)
recomm_norm = recomm_embed.divide(
    np.linalg.norm(recomm_embed.values, axis=1), axis="rows"
)
reddit_norm = reddit_embed.divide(
    np.linalg.norm(reddit_embed.values, axis=1), axis="rows"
)

# ### Create dimension generators (quite slow depending on size of embedding)

# content_dimgen = DimenGenerator(content_norm)
# recomm_dimgen = DimenGenerator(recomm_norm)
# reddit_dimgen = DimenGenerator(reddit_norm)

### Load saved Dimension generators
# unfortunately too large to share
with open(data_path("content_dimgen.pkl"), "rb") as handle:
    content_dimgen = pickle.load(handle)

with open(data_path("recomm_dimgen.pkl"), "rb") as handle:
    recomm_dimgen = pickle.load(handle)

with open(data_path("reddit_dimgen.pkl"), "rb") as handle:
    reddit_dimgen = pickle.load(handle)
```

```python
reddit_dim1_test = reddit_dimgen.generate_dimension_from_seeds(
    [(get_id("CNN"), get_id("Fox News"))]
)

reddit_auto_pairdf = pd.DataFrame(
    [reddit_dim1_test["left_comms"], reddit_dim1_test["right_comms"]],
    index=["left", "right"],
).T

content_dim1_test = content_dimgen.generate_dimension_from_seeds(
    [(get_id("CNN"), get_id("Fox News"))]
)

content_auto_pairdf = pd.DataFrame(
    [content_dim1_test["left_comms"], content_dim1_test["right_comms"]],
    index=["left", "right"],
).T

recomm_dim1_test = recomm_dimgen.generate_dimension_from_seeds(
    [(get_id("CNN"), get_id("Fox News"))]
)

recomm_auto_pairdf = pd.DataFrame(
    [recomm_dim1_test["left_comms"], recomm_dim1_test["right_comms"]],
    index=["left", "right"],
).T

recomm_dim_auto = dimdf_from_pairdf(recomm_auto_pairdf, recomm_norm)
reddit_dim_auto = dimdf_from_pairdf(reddit_auto_pairdf, reddit_norm)
content_dim_auto = dimdf_from_pairdf(content_auto_pairdf, content_norm)
```

```python
all_embeddings_auto = {
    "recomm": recomm_dim_auto,
    "reddit": reddit_dim_auto,
    "content": content_dim_auto,
    "reddit_avg": reddit_dim_normal,
}

scores = {k: tau_scorer(v.partisan) for k, v in all_embeddings_auto.items()}

plot_df_auto = pd.DataFrame(scores).rename_axis("agg").reset_index()
plot_df_auto = plot_df_auto.melt(id_vars=["agg"], var_name="embed", value_name="score")
plot_df_auto.to_csv(data_path("figures_in/dimgen_seeds_auto.csv"), index=False)

plot_df_auto
```

## Part 2: Using Manual Seeds

In this section, we use manually selected pairs of channels (one for all of the embeddings) instead of using the seeds which we previously obtained from the DimensionGenerators.

```python
left_channels = [
    "MeidasTouch",
    "The Young Turks",
    "Pod Save America",
    "Chapo Trap House",
    "Secular Talk",
    "The Humanist Report",
    "CNN",
    "Michael Moore",
]

right_channels = [
    "Fox News",
    "Project Veritas",
    "The Rubin Report",
    "Tim Pool",
    "Turning Point USA",
    "Donald J Trump",
    "BlazeTV",
    "CrowderBits",
    "AwakenWithJP",
    "Ben Shapiro",
]


# get channel id from title
left_ids = [get_id(c) for c in left_channels]
right_ids = [get_id(c) for c in right_channels]


def get_pair_df(normed_embed):
    # find pairing
    similarity = 1 + (
        normed_embed.loc[left_ids].values @ normed_embed.loc[right_ids].values.T
    )
    dist = -similarity
    row_ind, col_ind = linear_sum_assignment(dist)

    # pairs df
    pairdf = pd.DataFrame(
        zip(np.array(left_ids)[row_ind], np.array(right_ids)[col_ind]),
        columns=["left", "right"],
    )

    return pairdf


# pair the seeds from left and right using embedding distances (closest together)
pairdf = get_pair_df(recomm_norm)

# use subreddit pairs to get dimensions
recomm_dim_man = dimdf_from_pairdf(pairdf, recomm_norm)
reddit_dim_man = dimdf_from_pairdf(pairdf, reddit_norm)
content_dim_man = dimdf_from_pairdf(pairdf, content_norm)

all_embeddings = {
    "recomm": recomm_dim_man,
    "reddit": reddit_dim_man,
    "content": content_dim_man,
    "reddit_avg": reddit_dim_normal,
}

# compute rank correlations
scores = {k: tau_scorer(v.partisan) for k, v in all_embeddings.items()}
plot_df = pd.DataFrame(scores).rename_axis("agg").reset_index()

plot_df = plot_df.melt(id_vars=["agg"], var_name="embed", value_name="score")
plot_df.to_csv(data_path("figures_in/dimgen_seeds_manual.csv"), index=False)
plot_df
```

## Part 3: By training a classifier

In this part, we train classifiers which are provided the youtube channel embeddings from political channels as input, and have to output the political orientation of these channels.

These classifiers can then be used to obtain and real-valued scores by using the output probabilities instead of the labels.


#### Reading the political channel labels, removing center, setting every label to left or right

```python
classes = pd.read_csv(data_path("polarized_manoel_mbc_nondup.csv"))
classes["bias"] = (
    classes["bias"]
    .replace("extremeright", "right")
    .replace("centerleft", "left")
    .replace("centerright", "right")
    .replace("extremeleft", "left")
)
ordering = ["left", "right"]
classes = classes[classes.bias.isin(ordering)]

ordering_cat = pd.CategoricalDtype(ordering, ordered=True)
class_codes = classes.set_index("channelId")["bias"].astype(ordering_cat).cat.codes

embeddings = {
    "reddit": reddit_embed,
    "recomm": recomm_embed,
    "content": content_embed,
}
```

### Computing the classifier f1-scores, rank correlation of obtained social dimensions with our known labels

```python
f1_scores = defaultdict(list)
ord_scores = defaultdict(list)

N_FOLDS = 5

RETRIES = 100

# perform it many times since we have a small amount of channels
for seed in tqdm(range(RETRIES)):
    # we stratify on y, make sure we have a balanced set
    kf = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)

    for embed_name, embed in embeddings.items():
        joined_class = embed.join(class_codes.rename("bias"), how="inner")
        X = joined_class.drop(columns=["bias"])
        y = joined_class["bias"]

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            gbc = RandomForestClassifier()
            gbc.fit(X_train, y_train)

            pred = gbc.predict(X_test)

            # compute kendall tau using our classifiers probas
            ord_score = kendalltau_score(
                pd.Series(gbc.predict_proba(X_test)[:, 1], index=X_test.index),
                recomm_embed[[]]
                .join(classes.set_index("channelId"), how="inner")
                .reset_index()
                .iloc[test_index],
                ordering,
                id_col="channelId",
            )

            # compute kendall tau using our original social dimensions from subreddit vectors
            ord_reddit = kendalltau_score(
                reddit_dim.join(joined_class[[]], how="inner").partisan.iloc[
                    test_index
                ],
                recomm_embed[[]]
                .join(classes.set_index("channelId"), how="inner")
                .reset_index()
                .iloc[test_index],
                ordering,
                id_col="channelId",
            )

            # also compute f1
            f1 = f1_score(y_test, pred)

            f1_scores[embed_name].append(f1)
            ord_scores[embed_name].append(ord_score)
            ord_scores["reddit_avg"].append(ord_reddit)

# dont repeat it for all embeds
ord_scores["reddit_avg"] = ord_scores["reddit_avg"][: N_FOLDS * RETRIES]


# save dfs
ordering_score_df = pd.DataFrame(ord_scores).rename(
    columns={
        "content": "Content",
        "recomm": "Recommendation",
        "reddit": "Reddit",
        "reddit_avg": "Subreddit",
    }
)
ordering_score_df.to_csv(
    data_path("figures_in/ord_scores_train_label.csv"), index=False
)

f1_score_df = pd.DataFrame(f1_scores).rename(
    columns={
        "content": "Content",
        "recomm": "Recommendation",
        "reddit": "Reddit",
        "reddit_avg": "Subreddit",
    }
)
f1_score_df.to_csv(data_path("figures_in/f1_score_train_label.csv"), index=False)
```

### ECDF plot : comparison of distributions obtained from these classifiers vs the original dimension from subreddit vectors 

The dimensions we obtain tend to be distributed in a very sharp way: our classifier is often very sure that channels should be labelled as left or right, but we rarely see probabilities around 0.5, which is quite different from what we see overall in our original dimensions.

```python
j_class = (
    classes.set_index("channelId").join(reddit_embed[[]], how="inner").reset_index()
)

rdim_plot = (
    reddit_dim[["partisan"]]
    .rank(pct=True)
    .join(j_class.set_index("channelId"), how="inner")
)

preds_tes = defaultdict(list)
ys = defaultdict(list)

kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

for embed_name, embed in tqdm(embeddings.items()):
    joined_class = embed.join(class_codes.rename("bias"), how="inner")
    X = joined_class.drop(columns=["bias"])
    y = joined_class["bias"]

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        gbc = RandomForestClassifier()
        gbc.fit(X_train, y_train)

        preds_tes[embed_name].append(gbc.predict_proba(X_test)[:, 1])
        ys[embed_name].append(y_test)

ys_stack = {k: pd.concat(v) for k, v in ys.items()}
preds_stack = {k: np.hstack(v) for k, v in preds_tes.items()}

dfs_plot = []

for embed in embeddings:
    tmpdf = ys_stack[embed].to_frame("bias")
    tmpdf["partisan"] = preds_stack[embed]
    dfs_plot.append(tmpdf.replace({0: "left", 1: "right"}).assign(embed=embed))

dfs_plot.append(rdim_plot.assign(embed="Subbredit Dimensions"))

df_plot = pd.concat(dfs_plot).replace(
    {"reddit": "Reddit", "recomm": "Recommendation", "content": "Content"}
)

df_plot.to_csv(data_path("figures_in/ecdf_plot_df.csv"), index=False)
```

### Save the dimensions created by training the classifiers

Here, we train on the full set and save the resulting predictions.

```python
dimlabels = {}

for embed_name, embed in tqdm(embeddings.items()):
    # Ideally, we would not predict for the classes in training set - leave one out
    # overall it's not such a big deal, and dont really use this dataset (maybe 80/44000 channels)
    joined_class = embed.join(class_codes.rename("bias"), how="inner")
    X = joined_class.drop(columns=["bias"])
    y = joined_class["bias"]

    gbc = RandomForestClassifier()
    gbc.fit(X, y)

    preds = gbc.predict_proba(embed)
    dimlabels[embed_name] = pd.DataFrame(
        preds[:, 1], index=embed.index, columns=["partisan"]
    )

for k, v in dimlabels.items():
    v.reset_index().to_feather(
        data_path(f"dims/classification/{k}_partisan.feather.zstd"), compression="zstd"
    )
```

## Part 4: By training regressors on the original Reddit dimensions from subreddit vectors

This part is similar to part 3. However, instead of training classifiers from our ground-truth labels, we try to mimick our original social dimension scores by creating regressors, which are fed youtube channel embeddings as input.

We then compute the same rank correlation as before, to see if we're able to get similarly good (if not better) results, compared to our original dimensions.


### Train on full dataset, get rank correlations

```python
ordering_scores_reddit = {}

for emb_name, embed in tqdm(embeddings.items()):
    rfr = RandomForestRegressor(n_estimators=100, n_jobs=20)

    X = embed.drop(index=j_class.channelId)
    y = reddit_dim.loc[X.index].partisan.rank(pct=True)

    rfr.fit(X, y)

    preds = pd.Series(
        rfr.predict(embed.loc[j_class.channelId]), index=j_class.channelId
    )

    # compute rank correlation score
    ord_score = kendalltau_score(preds, j_class, ordering, id_col="channelId")

    ordering_scores_reddit[emb_name] = ord_score

# compute reddit dim ordering score to compare
ordering_scores_reddit["reddit_avg"] = kendalltau_score(
    reddit_dim.loc[j_class.channelId].partisan, j_class, ordering, id_col="channelId"
)

# save df

ordering_score_reddit_df = pd.DataFrame.from_dict(
    ordering_scores_reddit, orient="index"
).T

ordering_score_reddit_df.to_csv(
    data_path("figures_in/ordering_train_reddit.csv"), index=False
)
ordering_score_reddit_df
```

#### Compute average MAE (compared to reddit dim)

Here, we perform 5-fold cv, to check the MAE our regressors get, compared to the original scores.

Of course, we expect it to be low for the reddit embedding (since the original scores are computed from it).

```python
j_class = (
    classes.set_index("channelId").join(reddit_embed[[]], how="inner").reset_index()
)

mae_scores = defaultdict(list)

N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, random_state=0, shuffle=True)

ord_scores = defaultdict(list)

for emb_name, embed in embeddings.items():
    X = embed
    y = reddit_dim.partisan.rank(pct=True)

    for train_ind, test_ind in tqdm(kf.split(X), total=N_FOLDS):
        rfr = RandomForestRegressor(n_estimators=100, n_jobs=20)

        # train
        X_train = X.iloc[train_ind]
        X_test = X.iloc[test_ind]
        y_train = y.iloc[train_ind]
        y_test = y.iloc[test_ind]
        rfr.fit(X_train, y_train)

        preds = rfr.predict(X_test)

        mae_scores[emb_name].append(np.abs(preds - y_test).mean())


# save it
mae_score_reddit_df = pd.DataFrame(mae_scores)
mae_score_reddit_df.to_csv(data_path("figures_in/mae_train_reddit.csv"), index=False)
mae_score_reddit_df
```

### Save the dimensions created by training the regressors

Here, we train using 5-fold cv, and predict dimensions

```python
DIMS = ["partisan", "age", "affluence", "gender"]
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, random_state=0, shuffle=True)

for dim in tqdm(DIMS):
    dims = {}

    for emb_name, embed in embeddings.items():
        X = embed
        y = reddit_dim[dim]

        allpreds = []

        for train_ind, test_ind in tqdm(kf.split(X), total=N_FOLDS):
            rfr = RandomForestRegressor(n_estimators=100, n_jobs=20)

            # train
            X_train = X.iloc[train_ind]
            X_test = X.iloc[test_ind]
            y_train = y.iloc[train_ind]
            y_test = y.iloc[test_ind]
            rfr.fit(X_train, y_train)

            preds = pd.Series(rfr.predict(X_test), index=y_test.index)

            allpreds.append(preds)

        dims[emb_name] = pd.concat(allpreds)

    for k, v in dims.items():
        v.to_frame(dim).reset_index().to_feather(
            data_path(f"dims/regression_overall/{k}_{dim}.feather.zstd")
        )
```

## Extra : Training per category (except the few we are interested in)

```python
default_dims = pd.read_feather(
    data_path("dims/reddit.feather.zstd")
).set_index("channelId")
aggdf = pd.read_feather(data_path("per_channel_aggs.feather.zstd"))
topicdf = pd.read_json(data_path("id_to_topic.jsonl.gz"), lines=True)
aggdf["topic_id"] = aggdf["major_cat"].astype(int)
aggdf_topic = aggdf.merge(topicdf, on="topic_id").set_index("channelId")


DIMS = ["partisan", "age", "gender"]
CATS = ["News & Politics", "Music", "Howto & Style"]

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, random_state=0, shuffle=True)

for dim, cat in zip(tqdm(DIMS), CATS):
    dims = {}

    for emb_name, embed in embeddings.items():
        X = embed.join(aggdf_topic[aggdf_topic["topic"] == cat][[]], how="inner")
        y = reddit_dim[dim].loc[X.index]

        print(dim, cat, X.shape, y.shape)

        allpreds = []

        for train_ind, test_ind in tqdm(kf.split(X), total=N_FOLDS):
            rfr = RandomForestRegressor(n_estimators=100, n_jobs=20)

            # train
            X_train = X.iloc[train_ind]
            X_test = X.iloc[test_ind]
            y_train = y.iloc[train_ind]
            y_test = y.iloc[test_ind]
            rfr.fit(X_train, y_train)

            preds = pd.Series(rfr.predict(X_test), index=y_test.index)

            allpreds.append(preds)

        dims[emb_name] = pd.concat(allpreds)

    for k, v in dims.items():
        v.to_frame(dim).reset_index().to_feather(
            data_path(f"dims/regression_per_category/{k}_{dim}.feather.zstd")
        )
```

## Extra: Social dimensions Violin plot between known political channels and the rest

Here, we don't use the different computed dimensions, but plot the distribution of partisan-ness scores from our original social dimensions, for youtube channels which are known to be political, and the rest of unlabelled channels in our dataset.

We expect the partisan-ness score to be higher for known political channels.

```python
pness_normed = reddit_dim["partisan-ness"]

with_classes = (
    pness_normed.to_frame("pness")
    .join(classes.set_index("channelId"))
    .fillna("No-Label")
)
with_classes.reset_index().to_csv(data_path("figures_in/pness_labels.csv"))
```
