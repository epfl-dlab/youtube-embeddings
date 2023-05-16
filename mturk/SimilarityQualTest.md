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

# Mturk similarity experiment qualification test

In this notebook we read back the results from the similarity qualification test and grant the qualifications to the right mturk workers.

```python
import boto3
import numpy as np
import pandas as pd

%load_ext autoreload
%autoreload 2

# isort: off
import sys

sys.path += [".."]
# isort: on

from youtube_topics import data_path

qual_df = pd.read_csv(data_path(data_path("qual_test.csv")))
answ_df = qual_df[["WorkerId"]].assign(
    results=qual_df["Answer.batch-results"].apply(lambda s: np.array(s.split(",")))
)
valid_answer = np.array(["c", "c", "b", "c", "c", "c", "b", "a", "a", "b"])
```

## Check who got the right answers

```python
answ_df["perc"] = answ_df["results"].apply(lambda x: np.mean(x == valid_answer))
answ_df[answ_df["perc"] <= 0.2]
answ_df["perc"].plot(kind="hist")
```

```python
answ_df.query("perc <= 0.2")
```

```python
creds = pd.read_csv(data_path("new_user_credentials.csv"))
key_id, secret_key = creds.iloc[0][["Access key ID", "Secret access key"]]

client = boto3.client(
    "mturk", "us-east-1", aws_access_key_id=key_id, aws_secret_access_key=secret_key
)
```

### Reject workers who did very poor work

```python
assigns = answ_df.query("perc <= 0.4")["WorkerId"].values

for assign in assigns:
    try:
        client.reject_assignment(
            AssignmentId=assign,
            RequesterFeedback="Agreement with valid answer equal or lower than 20%",
        )
    except client.exceptions.RequestError:
        print(f"Assign failed for {assign}")
```

### Create qualification for good workers

```python
good_workers = answ_df.query("perc == 1")["WorkerId"].values
```

```python
resp = client.create_qualification_type(
    Name="youtube-qualtest",
    Keywords="Qualification,youtube",
    Description="Qualification for workers that passed the youtube similarity test",
    QualificationTypeStatus="Active",
    AutoGranted=True,
    AutoGrantedValue=0,
)
```

```python
for worker in good_workers:
    response = client.associate_qualification_with_worker(
        QualificationTypeId="36CIKHDZUHQGWDODEQ4R0GCJBEK0OW",
        WorkerId=worker,
        IntegerValue=1,
        SendNotification=False,
    )
```

```python
client.get_qualification_type(QualificationTypeId="36CIKHDZUHQGWDODEQ4R0GCJBEK0OW")
```

```python
res = client.list_workers_with_qualification_type(
    QualificationTypeId="36CIKHDZUHQGWDODEQ4R0GCJBEK0OW", MaxResults=100
)
```
