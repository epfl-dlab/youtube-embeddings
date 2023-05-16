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

# Creating the content embedding

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

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from youtube_topics import data_path
```

```python
logger = logging.getLogger()
logger.setLevel(logging.INFO)
```

## Load df containing metadata fetched from YouTube Data API

```python
df = pd.read_feather(data_path("filtered_channels_cropped.feather.zstd"))
```

### Compute SentenceTransformer embeddings

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

# apply sentence transformer model
desc_embeddings = model.encode(df["description"])
title_embeddings = model.encode(df["title"])

# map back to series
df["title_embed"] = list(title_embeddings)
df["desc_embed"] = list(desc_embeddings)

# compute mean per channel
agg_per_chan = df.groupby("channelId").agg(
    {"title_embed": np.mean, "desc_embed": np.mean}
)

# sum title and description embeddings
df_sum = pd.DataFrame(
    np.stack(agg_per_chan["title_embed"] + agg_per_chan["desc_embed"]),
    index=agg_per_chan.index,
)

# final semantic embedding
df_sum.reset_index().rename(columns=lambda i: str(i)).to_feather(
    data_path("summed_embedding.feather.zstd", compression="zstd")
)
```
