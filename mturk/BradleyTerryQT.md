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

# Bradley terry qualification test

In this notebook, we go throught the Bradley terry qualification test.

It is separated in two parts : creating the qualification test and reading back the results to give the Mturk qualifications.

```python
%load_ext autoreload
%autoreload 2
```

```python
import base64
import json

# isort: off
import sys

sys.path += [".."]
# isort: on

import boto3
import gspread
import innertube
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from youtube_topics import data_path
from youtube_topics.bradley_terry import answer_df, parse_answer
from youtube_topics.mturk import (df_to_sheet, mturkify, prep_mturk_batch,
                                  recent_thumbnails)
```

```python tags=[]
creds = pd.read_csv(data_path("new_user_credentials.csv"))
key_id, secret_key = creds.iloc[0][["Access key ID", "Secret access key"]]

client = boto3.client(
    "mturk", "us-east-1", aws_access_key_id=key_id, aws_secret_access_key=secret_key
)
```
# Part 1: Creating the qualification test 


## Selecting the channels for the qualification test

To select the channels, we simply take a random batch generated for a test bradley terry experiment.

We upload them to google drive via gsheet to select all the ones which are sort of obvious.

```python
# read batch
df = pd.read_csv(data_path("generated_batch.csv"))

# extract only the channels
all_pairs_df = pd.concat(
    pd.DataFrame(json.loads(base64.b64decode(x)))[["A_channelId", "B_channelId"]]
    for x in df["jsons"]
)

# write everything to the sheet
gc = gspread.service_account(filename="data/login-file.json")
df = all_pairs_df
client = innertube.InnerTube("WEB")
df = df.reset_index(drop=True)


def get_channels_summary(col):
    return df[col].apply(
        lambda chanid: recent_thumbnails(client, chanid, nthumbs=4)
        .head(4)
        .to_dict(orient="records")
    )


def process_title(s, col):
    return (
        s.apply(lambda l: [x["title"] for x in l])
        .apply(pd.Series)
        .rename(columns=lambda i: f"{col}_{i}")
    )


def process_thumb(s, col):
    return (
        s.apply(lambda l: [x["thumbnail"] for x in l])
        .apply(pd.Series)
        .rename(columns=lambda i: f"{col}_{i}")
        .applymap(lambda x: f'=IMAGE("{x}")')
    )


channel_summaries = [(col, get_channels_summary(col)) for col in tqdm(df.columns)]
df_thumb = pd.concat(
    [df] + [process_thumb(summ, col) for col, summ in channel_summaries], axis=1
)
df_title = pd.concat(
    [df] + [process_title(summ, col) for col, summ in channel_summaries], axis=1
)

fulldf = pd.concat((df_thumb, df_title)).sort_index(kind="mergesort")

# write to sheet
df_to_sheet(gc, all_pairs_df, "left")
```

## Reading back the channels, and setting it up for mturk 


### For the partisan experiment

```python
client = innertube.InnerTube("WEB")


def build_qualtest(qt_df, client):
    turkified = mturkify(client, qt_df)

    return prep_mturk_batch(turkified, batch_size=50)
```

```python
bt_qualtest_df = pd.read_csv(data_path("qualtest_bt_drive.csv"))

bt_qualtest_df_chans = (
    bt_qualtest_df.rename(columns={"A_channelId": "A", "B_channelId": "B"})[["A", "B"]]
    .sample(frac=1, replace=False, random_state=0)
    .reset_index(drop=True)
)

batched_partisan = build_qualtest(bt_qualtest_df_chans, client)

batched_partisan.to_csv(data_path("bradley_terry_qualtest.csv"), index=False)
```

### Similarly for gender, music

```python
gender_qualtest_df = (
    pd.read_csv(data_path("bt/bt_gender_howto_qt.csv"))
    .rename(columns={"masculine": "A", "feminine": "B"})[["A", "B"]]
    .sample(frac=1, replace=False, random_state=0)
    .reset_index(drop=True)
)

batched_gender = build_qualtest(gender_qualtest_df, client).assign(dim="gender")
```

```python
age_qualtest_df = (
    pd.read_csv(data_path("bt/bt_age_music_qt.csv"))
    .rename(columns={"young": "A", "old": "B"})[["A", "B"]]
    .sample(frac=1, replace=False, random_state=0)
    .reset_index(drop=True)
)

batched_age = build_qualtest(age_qualtest_df, client).assign(dim="age")
batched_age.to_csv(data_path("bt/bt_qualtest_age25.csv"), index=False)
```

# Part 2: Qualification test results


### For the age qualtest

```python
# true answer is always A, randomization is done client side on mturk
true_res = np.array(["A" for _ in range(25)])
results = pd.read_csv(data_path("bt/qualtest_age_redo_res.csv"))
results["answer"] = (
    results["Answer.batch-results"]
    .apply(parse_answer)
    .apply(lambda x: [y["age"] for y in x])
    .apply(np.array)
)
results["true"] = results["answer"].apply(lambda x: (x == true_res).mean())

# select all workers with one error or less
valid_workers = list(results.query("true > 0.95")["WorkerId"])

# create a qualification for all our workers which passed
resp = client.create_qualification_type(
    Name="youtube-agebt-qualtest",
    Keywords="Qualification,youtube,age",
    Description="Qualification for workers that passed the youtube bradley terry age test",
    QualificationTypeStatus="Active",
    AutoGranted=False,
)

# give them the qualification
for worker in valid_workers:
    response = client.associate_qualification_with_worker(
        QualificationTypeId=resp["QualificationType"]["QualificationTypeId"],
        WorkerId=worker,
        IntegerValue=1,
        SendNotification=False,
    )
```
