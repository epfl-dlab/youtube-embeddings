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

# Obtaining language information for youtube channels

Here, we use the various metadata we collected from youtube channels to attribute a channel language label.

We use metadata from : video title, description, caption language, and defaultAudioLanguage, defaultLanguage fields from the youtube data api.

```python
%load_ext autoreload
%autoreload 2
```

```python
# isort: off
import sys

sys.path += [".."]
# isort: on

from collections import Counter
from glob import glob

import emoji
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from whatthelang import WhatTheLang

from youtube_topics import data_path, read_proxies
from youtube_topics.inner_proxies.multiprocessing import compute_with_proxies
from youtube_topics.inner_proxies.video import get_caption
```

### Apply WhatTheLang on video title, description

Obtain language information using fasttext trained model (whatthelang) from title and description

```python
def remove_emoji(s):
    return emoji.replace_emoji(s, replace="")


wtl = WhatTheLang()


def apply_wtl(s):
    try:
        return wtl.predict_lang(s)
    except:
        return np.NaN


videos_df = pd.read_feather(
    data_path("filtered_channels_videos_flattened.feather.zstd")
)

seldf = videos_df[
    [
        "snippet.resourceId.videoId",
        "snippet.title",
        "snippet.description",
        "snippet.channelId",
    ]
]
seldf["desc_lang"] = seldf["snippet.description"].apply(remove_emoji).apply(apply_wtl)
seldf["title_lang"] = seldf["snippet.title"].apply(remove_emoji).apply(apply_wtl)

seldf.to_feather(data_path("per_video_metdata_lang.feather.zstd", compression="zstd"))
```

### Fetch caption language using innertube

Unofficial youtube api: fetch languages from yt with defaultAudioLanguage,defaultLanguage

```python
wrap_proxies = read_proxies(data_path("proxy_list.txt"))


def postprocess(iterable, res):
    return pd.concat(
        (iterable.reset_index(drop=True), pd.Series((x for x in res), name="result")),
        axis=1,
    )


labels = pd.read_feather(data_path("per_vid_language.feather.zstd"))
id_df = labels[
    labels[["snippet.defaultAudioLanguage", "snippet.defaultLanguage"]]
    .isna()
    .all(axis=1)
]["id"]
```

#### Compute captions, save in parts

```python
splits = np.array_split(id_df, 20)
for i, df in tqdm(enumerate(splits), total=len(splits)):
    res = compute_with_proxies(
        wrap_proxies,
        df,
        get_caption,
        postprocess,
        time_out=300,
        chunksize=50,
        njobs=50,
        delay=1,
    )

    pd.concat(res).reset_index(drop=True).to_feather(
        data_path(f"caption_languages/lang_{i:05d}.feather.zstd")
    )
```

#### Read back parts

```python
caption_lang = pd.concat(
    pd.read_feather(path)
    for path in glob(data_path("caption_languages/lang_*.feather.zstd"))
)

caption_lang["lang"] = caption_lang["result"].apply(
    lambda x: x[1] if x is not None else np.NaN
)
```

## Computing single language per channel


#### Read back defaultAudioLanguage, defaultLanguage

```python
default_lang = labels.rename(
    columns={
        "snippet.defaultAudioLanguage": "default_audio_language",
        "snippet.defaultLanguage": "default_language",
        "snippet.channelId": "channelId",
    }
)
```

#### Caption language from innertube

```python
capt_lang = caption_lang.rename(columns={"lang": "caption_language"})[
    ["id", "caption_language"]
]
```

#### title, desc languages from WhatTheLang

```python
metadata_lang = pd.read_feather(data_path("per_video_metdata_lang.feather.zstd"))
metadata_lang = metadata_lang.rename(columns={"snippet.resourceId.videoId": "id"})[
    ["id", "desc_lang", "title_lang"]
]
```

## Join them all

```python
merged = default_lang.merge(capt_lang, on="id", how="left").merge(
    metadata_lang, on="id", how="left"
)

merged_rep = merged.replace(
    {"en-US": "en", "en-GB": "en", "en-CA": "en", "CANT_PREDICT": np.NaN}
)
```

### Take language from first column, if none found, take it from the next, etc..

```python
cols = [
    "default_audio_language",
    "default_language",
    "caption_language",
    "title_lang",
    "desc_lang",
]

s = merged_rep[cols[0]]

for col in cols[1:]:
    s = np.where(pd.isna(s), merged_rep[col], s)

merged_rep["final_lang"] = s
```

### Addition : English vs All

Instead of weighing each language equally, we instead perform a lower-recall but higher precision method, which is to consider any video with any field pertaining to a channel other than english as a "OTHER" language video.

We then consider channels which have over 80% english videos (meaning less than 20% "OTHER" videos) as being English-speaking.

```python
ignore = (
    merged_rep[
        [
            "default_audio_language",
            "default_language",
            "caption_language",
            "desc_lang",
            "title_lang",
        ]
    ]
    .apply(lambda s: (s.notna()) & (s != "en") & (s != "zxx"))
    .any(axis=1)
)

merged_rep.loc[ignore, "final_lang"] = "ignore"
```

### Percentage of english

```python
language_counts = (
    merged_rep.groupby("channelId")["final_lang"]
    .apply(list)
    .apply(Counter)
    .to_frame("counter")
)

language_counts["en_percent"] = language_counts["counter"].apply(
    lambda c: c.get("en", 0) / sum(c.values())
)
```

### Considered en if > 0.8 english videos

```python
language_counts["label"] = "other"
language_counts.loc[language_counts["en_percent"] >= 0.8, "label"] = "english"
```

### Save

```python
# NEWER channel language dataframe, which performs ENGLISH vs REST approach

language_counts[["label"]].reset_index().to_feather(
    data_path("channel_language_final_with_ignore.feather.zstd", compression="zstd")
)
```

```python
# OLDER channel language dataframes, which perform any vs any choice

language_counts.reset_index().to_json(
    data_path("channel_language_final.jsonl.gz", lines=True, orient="records")
)

language_counts[["label"]].reset_index().to_feather(
    data_path("channel_language_final.feather.zstd", compression="zstd")
)
```
