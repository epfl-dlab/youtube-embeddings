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

# Fetching metadata from youtube channels using innertube

In this notebook, we perform similar work to the YoutubeFetching notebook but using the innertube api, which allows us to fetch everything without the need for an api key (and thus potentially no limits)

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

import logging

import innertube
import pandas as pd

from youtube_topics import data_path, read_proxies
from youtube_topics.inner_proxies.channel import get_playlist_vids
from youtube_topics.inner_proxies.multiprocessing import compute_with_proxies
```

### Reading the proxy list 

```python
wrap_proxies = read_proxies(data_path("proxy_list.txt"))
clients = [innertube.InnerTube("WEB", proxies=proxy) for proxy in wrap_proxies]
```

### Reading the DataFrame of channels from which we want to fetch playlist

```python
logging.getLogger().setLevel(logging.INFO)

df = pd.read_feather(data_path("filtered_channels.feather.zstd"))
df["playlistId"] = df["id"].apply(lambda x: "UU" + x[2:])
```

### Fetching playlists with proxies

```python
%%time


def postprocess(iterable, res):
    return pd.concat(
        (
            iterable.reset_index(drop=True),
            pd.Series(
                (
                    x[["videoId", "videoTitle", "playTime"]].to_dict(orient="records")
                    for x in res
                ),
                name="result",
            ),
        ),
        axis=1,
    )


res = compute_with_proxies(
    wrap_proxies,
    df["playlistId"],
    get_playlist_vids,
    postprocess,
    time_out=30,
    chunksize=10,
    njobs=50,
    delay=1,
)
```

#### Save results

```python
resdf = pd.concat(res).reset_index(drop=True)

resdf["filtered_vid_id"] = resdf["result"].apply(
    lambda l: [x["videoId"] for x in l if x["playTime"] > 60]
)

resdf.to_feather(data_path("channel_vid_id.feather.zstd"), compression="zstd")
```
