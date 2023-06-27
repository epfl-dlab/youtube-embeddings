---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Setting up the mturk similarity experiment

```python
%load_ext autoreload
%autoreload 2
```

```python
import innertube
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

# isort: off
import sys

sys.path += [".."]
# isort: on

import ast
import base64
import json

import numpy as np
import pandas as pd

from youtube_topics import data_path
from youtube_topics.mturk import (agreement_number, embed_closest,
                                  input_expected_result, prep_mturk_batch,
                                  read_mturk_res, sample_similar_mturk)
```

### Load embeddings

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

### Nearest neighbours

```python
reddit_neigh = NearestNeighbors(n_neighbors=100, metric="cosine").fit(reddit_embed)
content_neigh = NearestNeighbors(n_neighbors=100, metric="cosine").fit(content_embed)
recomm_neigh = NearestNeighbors(n_neighbors=100, metric="cosine").fit(recomm_embed)
```
### Sample similarity

```python
client = innertube.InnerTube("WEB")

embed_dict = {
    "recomm": (recomm_embed, recomm_neigh),
    "content": (content_embed, content_neigh),
    "reddit": (reddit_embed, reddit_neigh),
}

fracs = [0.01, 0.005, 0.0025]

used_channels = set()

# oversample since we want an exact length but some channels are removed
# might be better to do first pass, to check then second path instead
seed = 40
NUM_SAMPLES = 250
ACTUAL_SAMPLES = 100

dfs_mturk = {}

mturk_merged = []

for embed_name, (embed, neigh) in embed_dict.items():
    for frac in tqdm(fracs):
        print(embed_name, frac, seed)
        exp, shuf, mturked = sample_similar_mturk(
            embed,
            neigh,
            frac,
            client,
            seed=seed,
            num_samples=NUM_SAMPLES,
            ignore_channels=used_channels,
        )

        # mark channel as used, don't use it after
        used_channels |= set(exp["A"])

        # ignore those where A and B are not the closest, and where we dont have enough videos / deleted channel since collection
        mask = (exp.apply(embed_closest(embed), axis=1) == "AB") & (
            ~(mturked == "@@@None@@@").any(axis=1)
        )

        # filter them with mask
        exp = exp[mask].reset_index(drop=True).head(ACTUAL_SAMPLES)
        shuf = shuf[mask].reset_index(drop=True).head(ACTUAL_SAMPLES)
        mturked = mturked[mask].reset_index(drop=True).head(ACTUAL_SAMPLES)

        # make sure no nan, as many channels as we expect, same channels across sets
        assert len(exp) == (ACTUAL_SAMPLES)
        assert (
            len(exp) == len(shuf)
            and len(exp) == len(mturked)
            and (
                shuf.values
                == mturked[["A_channelId", "B_channelId", "C_channelId"]].values
            ).all()
        )

        seed += 1

        dfs_mturk[(embed_name, frac)] = (exp, shuf, mturked)

        # batch them for mturk
        df_prepped_mturk = prep_mturk_batch(mturked, batch_size=10).assign(
            frac=frac, embed=embed_name, seed=seed
        )

        mturk_merged.append(df_prepped_mturk)

full_df_mturk_large = pd.concat(mturk_merged).reset_index()

full_df_mturk_large.to_csv(data_path("similarity_experiment_mturk.csv"), index=False)

with open("similarity_experiment_mturk.pkl", "wb") as handle:
    import pickle

    pickle.dump(full_df_mturk_large, handle)
```

# Reading results from the mturk similarity experiment


### Load datasets and results

```python
recomm_embed = pd.read_feather(data_path("embeds/recomm.feather.zstd")).set_index(
    "channelId"
)
content_embed = pd.read_feather(data_path("embeds/content.feather.zstd")).set_index(
    "channelId"
)

reddit_embed = pd.read_feather(data_path("embeds/reddit.feather.zstd")).set_index(
    "channelId"
)

emb_dict = {"recomm": recomm_embed, "reddit": reddit_embed, "content": content_embed}

# contains 460 instead of expected 450 rows because some channels were removed from dataset
# so we re-created some hits so we have as many hits for each embed from the channels currently in the dataset
res_path = data_path("similarity/results_filtered.csv")

curr_mod_embed = read_mturk_res(res_path, cols=["HITId", "embed", "frac"])
curr_mod_embed["frac"] = curr_mod_embed["frac"].astype(str)
```

### Util to allow partial batches (not every sample using the same fraction)

```python
def explode_shared_batches(df, arr_colname):
    def split_fraction(s):
        evaled_frac = ast.literal_eval(s.frac)

        if isinstance(evaled_frac, list):
            # get res values
            res_arr = s[arr_colname]

            # keep unique fracs, in same order as presented
            _, indexes = np.unique(evaled_frac, return_index=True)
            unsorted_unique = [evaled_frac[index] for index in sorted(indexes)]

            # return split fractions
            return [
                pd.Series(
                    dict(
                        s.to_dict(),
                        **{
                            "frac": frac,
                            arr_colname: np.array(res_arr)[
                                [i for i, x in enumerate(evaled_frac) if x == frac]
                            ],
                        }
                    )
                )
                for frac in unsorted_unique
            ]

        return [pd.Series(s.to_dict())]

    return pd.concat(
        pd.DataFrame(x) for x in df.apply(split_fraction, axis=1)
    ).reset_index(drop=True)


def filter_mode(df):
    """Filter answers where mturker was able to answer more times than expected: people using extensions / bots"""
    lens = df["res"].apply(len)
    mode = lens.mode().iloc[0]
    return df[lens == mode]
```

### Get agreement across workers

```python
mode_mods = (
    curr_mod_embed.groupby(["HITId", "frac"])
    .apply(filter_mode)
    .reset_index(drop=True)
    .groupby(["HITId", "frac"])[["res"]]
    .apply(lambda d: pd.DataFrame(np.stack(d.iloc[:, 0])).mode(axis=0).iloc[0].tolist())
    .to_frame("mode_res")
)

agree_count = (
    curr_mod_embed.groupby(["HITId", "frac"])
    .apply(filter_mode)
    .reset_index(drop=True)
    .groupby(["HITId", "frac"])
    .apply(agreement_number("res"))
    .to_frame("agree_count")
)

mod0_embed = (
    read_mturk_res(
        res_path,
        cols=["HITId", "embed", "Input.jsons", "frac"],
    )
    .groupby(["HITId", "frac"])
    .apply(filter_mode)
    .reset_index(drop=True)
)

embed_with_input = pd.concat((mod0_embed,)).drop_duplicates(subset=["HITId"])
embed_with_input["frac"] = embed_with_input["frac"].astype(str)
```

### Comparing answers on the other datasets

```python
mode_mods = explode_shared_batches(mode_mods.reset_index(), "mode_res").set_index(
    ["HITId", "frac"]
)
agree_count = explode_shared_batches(
    agree_count.reset_index(), "agree_count"
).set_index(["HITId", "frac"])

embed_input_full = pd.concat(
    embed_with_input[["HITId", "embed", "frac"]]
    .rename(columns={"embed": "embed_data"})
    .assign(
        embed=embed,
        exp_result=embed_with_input.apply(
            lambda row: input_expected_result(row["Input.jsons"], emb_dict[embed]),
            axis=1,
        ),
    )
    for embed in emb_dict.keys()
)

embed_input_full = explode_shared_batches(embed_input_full, "exp_result")


full_df_agreement = (
    embed_input_full.set_index(["HITId", "frac"])[["embed_data", "embed", "exp_result"]]
    .join(agree_count)
    .join(mode_mods)
    .reset_index()
)

full_df_agreement["frac"] = full_df_agreement["frac"].astype("float")
```

<!-- #region tags=[] -->
### Save
<!-- #endregion -->

```python
full_df_agreement.to_json(
    data_path("figures_in/mturk_similarity_results_filtered.jsonl"),
    lines=True,
    orient="records",
)
```

## Appendix: filtering comparisons

Some of the comparisons (16/900) were performed on a channels which are no longer in the dataset.

To make sure we had exactly as many comparisons with channels in our dataset for each embedding, we re-did as many comparisons, and filter out the previous ones.

This is the difference between "mturk_similarity_results_filtered", and "mturk_similarity_results_final"

```python
def filter_json_channels(x):
    df = pd.DataFrame(json.loads(base64.b64decode(x["Input.jsons"])))

    # only keep if all channels are in the dataset
    mask = (
        df[["A_channelId", "B_channelId", "C_channelId"]]
        .apply(lambda x: x.isin(reddit_embed.index))
        .all(axis=1)
    )
    df_filt = df[mask]

    length = x["Input.length"]

    # update array, unless size was already wrong
    arr = np.array(x["Answer.batch-results"].split(","))
    res = x["Answer.batch-results"] if len(arr) != length else ",".join(arr[mask])

    return (
        base64.b64encode(df_filt.to_json(None, orient="records").encode()).decode(),
        mask.sum(),
        res,
    )


df_sim = pd.read_csv(data_path("similarity/results.csv"))

df_sim["Input.jsons"], df_sim["Input.length"], df_sim["Answer.batch-results"] = list(
    zip(*df_sim.apply(filter_json_channels, axis=1))
)

df_sim.to_csv(data_path("similarity/results_filtered.csv"), index=False)
```
