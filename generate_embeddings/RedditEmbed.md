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

# Setting up the reddit embedding from reddit shares

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

import pandas as pd

from youtube_topics import data_path
from youtube_topics.preprocessing import filter_spammy_authors, gpby_counts
from youtube_topics.reddit_averaging import (averaging_embed, get_scores_df,
                                             read_seed_dim_dict,
                                             video_stratify)
```

## Read channel, subreddit dataframe

```python
subs_df = pd.read_parquet(data_path("submissions_with_channel.parquet/"))
comms_df = pd.read_parquet(
    data_path(
        "/dlabdata1/scratch_060/scratch/boesinge/youtube-topics/comments_with_channel.parquet"
    )
)

concat = pd.concat((comms_df, subs_df))
```

#### Save dfs for plotting

```python
author_mentions = concat.groupby("author").size()
# author_mentions.to_frame('count').reset_index().to_feather('author_mentions.feather.zstd', compression='zstd')

unique_mentions_per_chan = concat.groupby("channelId").size()
unique_subs_per_chan = concat.groupby("channelId")["subreddit"].nunique()

# unique_mentions_per_chan.to_frame('mentions').reset_index().to_feather('channel_mentions.feather.zstd', compression='zstd')
# unique_subs_per_chan.to_frame('subreddits').reset_index().to_feather('channel_subreddits.feather.zstd', compression='zstd')
```

### Remove spammy authors (> 1000 posts), group by channel, subreddit get counts

```python
comms_df_filtered = filter_spammy_authors(comms_df)
subs_df_filtered = filter_spammy_authors(subs_df)

comms_df_gb = gpby_counts(comms_df_filtered)
subs_df_gb = gpby_counts(subs_df_filtered)
```

## Averaging reddit


#### Read subreddit vectors, subreddit seeds

```python
embed_meta_old = pd.read_csv(data_path("embedding-metadata.tsv"), sep="\t")
embed_vec_old = pd.read_csv(
    data_path("data/embedding-vectors.tsv"), sep="\t", header=None
)
embed_vec_old.index = embed_meta_old.community

old_seed_dim_dict = read_seed_dim_dict(data_path("social_dimensions.yaml.txt"))
```

#### Merge posts and comments, keep only subreddits for which we have subreddit vectors

```python jupyter={"outputs_hidden": true}
merged_df = pd.concat((comms_df_gb, subs_df_gb)).reset_index(drop=True)
merged_df_old = merged_df[merged_df.subreddit.isin(embed_vec_old.index)]

merged_df_old_summed = (
    merged_df_old.groupby(["subreddit", "channelId"])["total"].sum().reset_index()
)

## save them to feather

# merged_df_old_summed.to_feather(
#     "comments_overlap_old_subreddits.feather.zstd", compression="zstd"
# )

## to read them
merged_df_old_summed = pd.read_feather(
    data_path("comments_overlap_old_subreddits.feather.zstd")
)
```

### Creating the embeddings

```python
embeds_dict = {}

embeds_dict["default"] = averaging_embed(
    merged_df_old_summed, embed_vec_old, pmi=False, shrinkage=False
)
embeds_dict["pmi"] = averaging_embed(
    merged_df_old_summed, embed_vec_old, pmi=True, shrinkage=False
)
embeds_dict["shrinkage"] = averaging_embed(
    merged_df_old_summed, embed_vec_old, pmi=False, shrinkage=True
)
embeds_dict["pmi_shrinkage"] = averaging_embed(
    merged_df_old_summed, embed_vec_old, pmi=True, shrinkage=True
)
```

### Video Stratify embeddings

```python
embeds_dict["strat_log"] = video_stratify(
    merged_df_old_summed,
    embed_vec_old,
    pmi=False,
    shrinkage_channel=True,
    shrinkage_video=True,
    weight_func="log",
)
embeds_dict["strat_equal"] = video_stratify(
    merged_df_old_summed,
    embed_vec_old,
    pmi=False,
    shrinkage_channel=True,
    shrinkage_video=True,
    weight_func="equal",
)
```

### Computing the dims

```python
dims_dict = {
    k: get_scores_df(old_seed_dim_dict, v, embed_vec_old)[1]
    for k, v in embeds_dict.items()
}
```

#### Save dims and embeds

```python
def save_embed_feather(embed, name, index_name="channelId"):
    embed.rename_axis(index_name).reset_index().rename(
        columns=lambda i: str(i)
    ).to_feather(data_path(f"{name}.feather.zstd"), compression="zstd")


for name, embed in embeds_dict.items():
    save_embed_feather(embed, f"data/embeds/{name}")

for name, dims in dims_dict.items():
    save_embed_feather(dims, f"data/dims/{name}")
```
