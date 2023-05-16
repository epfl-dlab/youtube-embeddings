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
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from youtube_topics import data_path
from youtube_topics.bradley_terry import answer_df
from youtube_topics.mturk import (df_to_sheet, mturkify, prep_mturk_batch,
                                  recent_thumbnails)
```

# Part 1: Creating the qualification test 


## Selecting the channels for the qualification test

To select the channels, we simply take a random batch generated for a test bradley terry experiment.

We upload them to google drive via gsheet to select all the ones which are sort of obvious.

```python
# read batch
df = pd.read_csv(data_path("newest_test_bt_feb7.csv"))

# extract only the channels
all_pairs_df = pd.concat(
    pd.DataFrame(json.loads(base64.b64decode(x)))[["A_channelId", "B_channelId"]]
    for x in df["jsons"]
)

# write everything to the sheet
gc = gspread.service_account(filename="data/reddit-rules-b2ab6d75ef7d.json")
df = all_pairs_df
client = innertube.InnerTube("WEB")
df = df.reset_index(drop=True)

def get_channels_summary(col):
    return df[col].apply(lambda chanid: recent_thumbnails(client, chanid, nthumbs=4).head(4).to_dict(orient="records"))

def process_title(s, col):
    return s.apply(lambda l: [x["title"] for x in l]).apply(pd.Series).rename(columns=lambda i: f"{col}_{i}")
def process_thumb(s, col):
    return s.apply(lambda l: [x["thumbnail"] for x in l]).apply(pd.Series).rename(columns=lambda i: f"{col}_{i}").applymap(lambda x: f'=IMAGE("{x}")')

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

```python
bt_qualtest_df = pd.read_csv(data_path("qualtest_bt_drive.csv"))

bt_qualtest_df_chans = (
    bt_qualtest_df.rename(columns={"A_channelId": "A", "B_channelId": "B"})[["A", "B"]]
    .sample(frac=1, replace=False, random_state=0)
    .reset_index(drop=True)
)

client = innertube.InnerTube("WEB")

turkified = mturkify(client, bt_qualtest_df_chans)

batched = prep_mturk_batch(turkified, batch_size=50)

batched.to_csv(data_path("bradley_terry_qualtest.csv"), index=False)
```

# Part 2: Qualification test results


## Check who got the right answers

```python
# using answer_df to add answer column is dirty, todo fix

# read true answers
true_res = pd.read_csv(data_path("bradley_terry_qualtest_res.csv"))
answer_df(true_res)
true_res = true_res.answers.apply(lambda x: np.array([y["partisan"] for y in x])).iloc[
    0
]

# read user answer
results = pd.read_csv(data_path("bradley_terry_qualtest_results.csv"))
try:
    answer_df(results)
except:
    pass
results = results[results["answers"].apply(len) == 25]
results["answers"] = results.answers.apply(
    lambda x: np.array([y["partisan"] for y in x])
)

results["true"] = res.apply(lambda x: (x == true_res).mean())

valid_workers = results.query("true > 0.8").WorkerId

results["true"].plot(kind="hist")
```

```python
creds = pd.read_csv(data_path("new_user_credentials.csv"))
key_id, secret_key = creds.iloc[0][["Access key ID", "Secret access key"]]

client = boto3.client(
    "mturk", "us-east-1", aws_access_key_id=key_id, aws_secret_access_key=secret_key
)
```

### Create qualification for good workers

```python
resp = client.create_qualification_type(
    Name="youtube-politicalbt-qualtest",
    Keywords="Qualification,youtube,political",
    Description="Qualification for workers that passed the youtube bradley terry political test",
    QualificationTypeStatus="Active",
    AutoGranted=True,
    AutoGrantedValue=0,
)
```

```python
for worker in valid_workers:
    response = client.associate_qualification_with_worker(
        QualificationTypeId=resp["QualificationType"]["QualificationTypeId"],
        WorkerId=worker,
        IntegerValue=1,
        SendNotification=False,
    )
```

```python
client.get_qualification_type(QualificationTypeId="3KPNUZIAIXR6BLN86WD1UCYS4KDH1T")
```
