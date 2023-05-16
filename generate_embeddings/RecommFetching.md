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

# Fetching recommendations from innertube and setting up the recommendation graph

The innertube api allows us to obtain recommendations for any video

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

import pandas as pd
from youtube_topics.inner_proxies.multiprocessing import compute_with_proxies, retry
from youtube_topics.inner_proxies.recommendation import RecommendationExtractor
from tqdm.auto import tqdm

from youtube_topics import data_path, read_proxies
```

### Proxy list

```python
wrap_proxies = read_proxies(data_path("proxy_list.txt"))
```

### Fetching recommended videos

```python
# should be equivalent to previous formulation


@retry(tries=3, delay=10)
def recommended_videos(client, vid_id):
    gen = RecommendationExtractor.generator(client, vid_id)
    retval = next(gen)

    return retval
```

### DataFrame of channels to fetch

```python
df = pd.read_feather(data_path("channel_vid_id.feather.zstd"))
exploded = df.explode("filtered_vid_id")
exploded = exploded.assign(channelId="UC" + exploded.playlistId.apply(lambda x: x[2:]))[
    ["channelId", "filtered_vid_id"]
]
```

### Fetching process with proxies

```python
logging.basicConfig(filename="recommendation.log", encoding="utf-8", level=logging.INFO)


def postprocess(iterable, res):
    return pd.concat(
        (
            iterable.reset_index(drop=True),
            pd.Series((x.to_dict(orient="records") for x in res), name="result"),
        ),
        axis=1,
    )


for i in tqdm(range(0, 100)):
    sample = (
        exploded.groupby("channelId")
        .sample(1, random_state=i)
        .dropna()
        .set_index("channelId")["filtered_vid_id"]
    )

    res = compute_with_proxies(
        wrap_proxies,
        sample,
        recommended_videos,
        postprocess,
        time_out=30,
        chunksize=10,
        njobs=50,
        delay=1,
    )

    resdf = pd.concat(res).reset_index(drop=True)

    resdf.to_feather(
        data_path(f"innertube_recomms2/id_{i:02d}_df.feather.zstd"), compression="zstd"
    )

    del res, resdf
```
