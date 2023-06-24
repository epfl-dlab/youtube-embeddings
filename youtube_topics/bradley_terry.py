from typing import Any, List
import regex as re
from itertools import chain
import choix
import scipy
import plotly.express as px

import logging
import matplotlib.pyplot as plt
import json
import base64
import pandas as pd
import numpy as np

import time
import boto3
import innertube
import seaborn as sns
from plotly.offline import plot

from youtube_topics import data_path
from youtube_topics.mturk import channel_df, mturkify, prep_mturk_batch

from collections import defaultdict

client = innertube.InnerTube("WEB")


def parse_answer(s: str) -> List[Any]:
    """Parse json-ish encoded answer from mturk bt experiment

    Args:
        s (str): the json string

    Returns:
        List[Any]: decoded answer
    """
    rem_slash = s.replace("\\", "")[1:-1]
    split = re.split(r"(?<=}),(?={)", rem_slash, flags=re.VERSION1)
    return [json.loads(x) for x in split]


# extracting channelId
def readjson(s):
    return json.loads(base64.b64decode(s))
def char_chan(char):
    return lambda series: series.apply(lambda arr: [x[f"{char}_channelId"] for x in arr])
a_chan = char_chan("A")
b_chan = char_chan("B")


# get mode answer per batch
def agg_mode(x):
    stacked = np.stack(x)
    return pd.DataFrame(stacked).mode().values.ravel()


# return channelId instead of A/B, zip(win, lose)
def channel_where(key: str):
    def channel_where_(row):
        wins = np.where(row[key] == "A", row["A"], row["B"])
        loses = np.where(row[key] == "A", row["B"], row["A"])

        return list(zip(wins, loses))

    return channel_where_


def sample_tuples(num_pairs, all_nodes, seed):
    """Sample pairs of elements of all_nodes
    Each pair is sampled num_pairs times, and each element can only be paired once with each other element
    """

    s = pd.Series([num_pairs for _ in all_nodes], index=all_nodes)

    tupmin = defaultdict(set)
    tupmax = defaultdict(set)

    # at each iter, sample one single channel
    # match it with the channel which has been associated the least amount of times
    while s.sum() > 0:
        n = s[s > 0].sample(1, random_state=seed).index[0]

        selection = s[
            (s.index != n)
            & (s > 0)
            & (~s.index.isin(tupmax[n]))
            & (~s.index.isin(tupmin[n]))
        ]

        m = selection.sort_values(ascending=False).index[0]

        min_, max_ = min(n, m), max(n, m)
        tupmin[min_] |= {max_}
        tupmax[max_] |= {min_}

        s.loc[m] -= 1
        s.loc[n] -= 1

    all_tuples = [(a, b) for a, tuples in tupmin.items() for b in tuples]

    return all_tuples


def answer_df(fulldf, dimensions=["partisan"]):
    """Apply to mturk results dataframe to obtain parsed results

    Args:
        fulldf (pd.DataFrame): Mturk results DataFrame (from reading csv)

    Returns:
        pd.DataFrame: parsed results
    """

    # parse answer
    fulldf = fulldf[fulldf['AssignmentStatus'] != 'Rejected']
    fulldf["answers"] = fulldf["Answer.batch-results"].apply(parse_answer)

    def key_to_list(d):
        return pd.Series(([x[key] for x in d] for key in dimensions), index=dimensions)

    # get one col per answer (political, partisan)
    answerdf = pd.concat(
        (fulldf[["HITId"]], fulldf["answers"].apply(key_to_list).applymap(np.array)),
        axis=1,
    )

    # read channels
    jsons = fulldf["Input.jsons"].apply(readjson)
    answerdf["A"] = a_chan(jsons)
    answerdf["B"] = b_chan(jsons)

    # group per hit, apply mode
    per_hit = answerdf.groupby("HITId").agg(
        dict({"A": "first", "B": "first"}, **{k: agg_mode for k in dimensions})
    )

    # create list of (winner, loser) for each question with channel ids
    for k in dimensions:
        per_hit[f"{k}_tup"] = per_hit.apply(channel_where(k), axis=1)

    return per_hit


def plot_dim_bins(df, xs, ys, xcol="dimnorm", ycol="nessnorm", title_col='channelTitle'):
    """Plot dimension dataframe with horizontal and vertical lines representing bins"""
    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        color="topic",
        marginal_x="histogram",
        marginal_y="violin",
        hover_data=["label", title_col],
    )

    for x in xs:
        fig.add_vline(x)

    for y in ys:
        fig.add_hline(y)

    return fig


def sample_per_label(df, num_per_label, oversample=1, seed=0):
    """Hierarchical sampling over labels"""
    
    sampled_channels = df.groupby("label").apply(lambda x: x.sample(
        min(len(x), int(np.ceil(num_per_label * oversample))), replace=False, random_state=seed
    )).reset_index(level=0, drop=True)

    mask = (
            sampled_channels.reset_index().channelId.apply(
                lambda x: channel_df(client, x).dropna().shape[0]
            )
            == 1
        )

    masked = sampled_channels.reset_index()[mask]

    final_df = (
        masked.groupby("label")
        .apply(lambda x: x[:num_per_label])
        .reset_index(drop=True)
        .set_index("channelId")
    )

    if not (final_df['label'].value_counts() == num_per_label).all():
        logging.error(f'Could not sample {num_per_label} channels for all labels, try increasing oversample, or lower it')
        
    return final_df


def sample_bt(df, sample_size, pairings_per_channel, seed=1, batch_size=50):
    """Sample channels from embedding, setup Bradley Terry experiment for mturk"""

    # sample
    all_channels = pd.Series(df.index.values)
    sampled_100 = all_channels.sample(
        n=sample_size, random_state=seed, replace=False
    ).reset_index(drop=True)

    # turkify
    turkified_channels = mturkify(client, sampled_100.to_frame("A"))
    turkified_channels.index = sampled_100

    # create tournament
    sampled_tuples = sample_tuples(pairings_per_channel, sampled_100, seed)

    # tournament df
    channel_pairs = pd.DataFrame(sampled_tuples, columns=["A", "B"]).sample(
        frac=1, replace=False, random_state=seed
    )

    # prep mturk df
    A_info = turkified_channels.loc[channel_pairs["A"]].reset_index(drop=True)
    B_info = (
        turkified_channels.loc[channel_pairs["B"]]
        .rename(columns=lambda s: s.replace("A", "B"))
        .reset_index(drop=True)
    )
    mturked_df = pd.concat((A_info, B_info), axis=1)
    batched_mturked_bt = prep_mturk_batch(mturked_df, batch_size=batch_size)
    batched_mturked_bt = batched_mturked_bt.assign(batch=range(len(batched_mturked_bt)))

    return batched_mturked_bt


def topic_refetch(df, topic, xs, ys, dim, mean_method='mean', std_method='std'):
    """Attribute label for each channel in topic based on dimension and dimension-ness value"""
    
    topiconly = df.query("topic == @topic")
    dimness = f"{dim}-ness"

    topiconly = topiconly.assign(
        topicdim=(topiconly[dim] - topiconly[dim].agg(mean_method)) / topiconly[dim].agg(std_method),
        topicness=(topiconly[dimness] - topiconly[dimness].agg(mean_method))
        / topiconly[dimness].agg(std_method),
    )

    topiconly["label"] = 0
    len(ys) - 1
    lenx = len(xs) - 1
    for i, (x1, x2) in enumerate(zip(xs, xs[1:])):
        for j, (y1, y2) in enumerate(zip(ys, ys[1:])):
            n = i + (1 - j) * (lenx)

            inds = topiconly.query(
                "(@x1 <= topicdim) and (topicdim < @x2) and (@y1 < topicness) and (topicness < @y2)"
            ).index

            topiconly.loc[inds, "label"] = n

    return topiconly


def get_bt_topic(
    df, n_per_label, pair_per_channel, seed=1, batch_size=50, oversample=1):
    """Setup bradley terry samples for topic"""
    
    sampled = sample_per_label(df, n_per_label, seed=seed, oversample=oversample)
    return sample_bt(sampled, len(sampled), pair_per_channel, batch_size=batch_size)


def dim_corrs(rankdf, dims, dim="partisan"):
    """Compute kendalltau rank correlation between Bradley terry ranks and dimension ranks"""
    rank_reddit = (
        (dims.rename(columns={"partisan-ness": "political"}))
        .loc[rankdf.index][[dim]]
        .apply(lambda x: x.argsort().argsort())
    )

    fullrank = rankdf.join(rank_reddit.rename(columns=lambda i: f"reddit_{i}"))

    return scipy.stats.kendalltau(fullrank[dim], fullrank[f"reddit_{dim}"], variant="c").correlation


def plot_reg_correlation(rankdf, dim, default_dims, ax=None,
                         saved_dims_path='dims/regression_overall',
                         container_label=True,
                         legend=True,
                         use_hue=True,
                         order=None,
                         x_label=True,
                         y_label=True,
                         replace_dict=None
                        ):
    """Plot correlation between Bradley terry ranks and dimension ranks obtained
    from regression training on our baseline reddit dimensions
    """
    if ax is None:
        fig, ax = plt.subplots()

    recomm_reg = pd.read_feather(
        data_path(f"{saved_dims_path}/recomm_{dim}.feather.zstd")
    ).set_index("channelId")
    # reddit_reg = pd.read_feather(
    #     data_path(f"{saved_dims_path}/reddit_{dim}.feather.zstd")
    # ).set_index("channelId")
    reddit_reg = default_dims
    content_reg = pd.read_feather(
        data_path(f"{saved_dims_path}/content_{dim}.feather.zstd")
    ).set_index("channelId")

    dims_dict = {"recomm": recomm_reg, "reddit": reddit_reg, "content": content_reg}
    
    results_df = (
        pd.DataFrame(
            {
                ind: {k: dim_corrs(rankdf, v, dim=ind) for k, v in dims_dict.items()}
                for ind in [dim]
            }
        )
        .rename_axis("embed")
        .reset_index()
    )
    
    if replace_dict is not None:
        results_df = results_df.replace(replace_dict)
        
    if use_hue:
        results_df = results_df.melt(id_vars="embed")
    
    sns.barplot(
        results_df,
        y="value" if use_hue else dim,
        x="variable" if use_hue else 'embed',
        hue="embed" if use_hue else None,
        hue_order=["content", "recomm", "reddit"] if use_hue else None,
        order=[dim] if use_hue else order,
        ax=ax,
    )

    if container_label:
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.2f")

    handles, labels = ax.get_legend_handles_labels()

    ax.set(title="Regression train correlation vs baseline")
    if not y_label:
        ax.set(ylabel=None)
    if not x_label:
        ax.set(xlabel=None)
        
    return results_df

#     line = ax.axhline(dim_corrs(rankdf, default_dims, dim=dim), c="r", label="reddit baseline")

#     if legend:
#         ax.legend(handles + [line], labels + [line.get_label()], loc='lower right')


def rank_df(answerdf, dimensions=["partisan"]):
    """Get rank dataframe by applying bradley terry model to answerdf

    Args:
        answerdf (pd.DataFrame): df obtained from answer_df function

    Returns:
        pd.DataFrame: The rank dataframe
    """

    all_items = sorted(
        np.unique(list(chain.from_iterable(answerdf[["A", "B"]].values.ravel())))
    )
    n_items = len(all_items)
    cat = pd.CategoricalDtype(all_items, ordered=True)

    ranks = {}

    for key in dimensions:
        chained_key = list(chain.from_iterable(answerdf[f"{key}_tup"]))
        as_ind = [
            tuple(x)
            for x in pd.DataFrame(chained_key)
            .astype(cat)
            .apply(lambda s: s.cat.codes)
            .values
        ]

        key_choix = choix.ilsr_pairwise(n_items, as_ind, alpha=0.01)
        key_rank = key_choix.argsort().argsort()

        ranks[key] = key_rank

    df_end = pd.DataFrame(
        [all_items] + list(ranks.values()), index=["channelId"] + list(ranks.keys())
    ).T

    return df_end


def get_rank(res, dimension, reverse_dim=True):
    """Compute Bradley terry rank from mturk answers df"""
    answers = answer_df(res, dimensions=[dimension])
    rank = rank_df(answers, dimensions=[dimension]).set_index("channelId")
    return rank.applymap(lambda x: len(rank) - x) if reverse_dim else rank


def plot_res_scatter(path, outpath, dimension, title_series,
                     title_col='channelTitle', saved_dims_path='dims/regression_overall',
                     reverse_dim=True):
    """Plot correlation scatter plot between BT ranks and dimension ranks"""
    
    df = pd.read_csv(path)
    df_rank = get_rank(df, dimension, reverse_dim=reverse_dim)

    recomm_reg = pd.read_feather(
        data_path(f"{saved_dims_path}/recomm_{dimension}.feather.zstd")
    ).set_index("channelId")

    tmpdf = df_rank.join(title_series).join(
        recomm_reg.rename(columns={dimension: f"{dimension}_recomm"})
    )
    df_res = tmpdf.assign(
        **{f"{dimension}_recomm": tmpdf[f"{dimension}_recomm"].rank()}
    )
    df_res["url"] = pd.Series(df_res.index, index=df_res.index).apply(
        lambda x: f"https://youtube.com/channel/{x}"
    )

    fig = px.scatter(
        df_res, x=dimension, y=f"{dimension}_recomm", hover_data=[title_col, "url"]
    )

    # Get HTML representation of plotly.js and this figure
    plot_div = plot(fig, output_type="div", include_plotlyjs=True)

    # Get id of html div element that looks like
    # <div id="301d22ab-bfba-4621-8f5d-dc4fd855bb33" ... >
    res = re.search('<div id="([^"]*)"', plot_div)
    div_id = res.groups()[0]

    # Build JavaScript callback for handling clicks
    # and opening the URL in the trace's customdata
    js_callback = """
    <script>
    var plot_element = document.getElementById("{div_id}");
    plot_element.on('plotly_click', function(data){{
        var point = data.points[0];
        if (point) {{
            window.open(point.customdata[1]);
        }}
    }})
    </script>
    """.format(
        div_id=div_id
    )

    # Build HTML string
    html_str = """
    <html>
    <body>
    {plot_div}
    {js_callback}
    </body>
    </html>
    """.format(
        plot_div=plot_div, js_callback=js_callback
    )

    # Write out HTML file
    with open(outpath, "w") as f:
        f.write(html_str)


def plot_corr_csv(path, dimension, default_dims, ax=None, reverse_dim=True, **kwargs):
    """Load mturk csv results and plot correlation with our regression
    trained dimension ranks"""
    
    df = pd.read_csv(path)
    df_rank = get_rank(df, dimension, reverse_dim=reverse_dim)

    return plot_reg_correlation(df_rank, dimension, default_dims, ax=ax, **kwargs)

    
class BTMTurkHelper(object):

    def __init__(self, secret_path, queue_url, limit_qual_id, counter_qual_id, counter_limit=4, endpoint='https://mturk-requester-sandbox.us-east-1.amazonaws.com'):

        with open(data_path(secret_path), 'r') as handle:
            handle.readline()
            user, pwd = handle.readline().strip().split(',')

        self.counter_qual_id = counter_qual_id
        self.limit_qual_id = limit_qual_id
        self.counter_limit = counter_limit
        
        self.queue_url = queue_url
        self.mturk = boto3.client('mturk',
                                  aws_access_key_id=user,
                                  aws_secret_access_key=pwd,
                                  region_name='us-east-1',
                                  endpoint_url=endpoint)

        self.sqs = boto3.client('sqs',
                                aws_access_key_id=user,
                                aws_secret_access_key=pwd,
                                region_name='us-east-1')

    def increment_user_qualification(self, worker_id):
        response = None

        # get counter score
        try:
            response = self.mturk.get_qualification_score(
            QualificationTypeId=self.counter_qual_id,
            WorkerId=worker_id
            )
        # ignore error if not yet set
        except self.mturk.exceptions.RequestError as e:
            if not e.response['Error']['Message'].startswith('You requested a Qualification that does not exist.'):
                logging.error(f'Could not get qualification {self.counter_qual_id} for worker {worker_id}')
                
        # increment counter score
        # if not yet set, set it to 1
        incr = (response['Qualification']['IntegerValue'] if response is not None else 0) + 1 
        
        print('Increasing counter')
        self.mturk.associate_qualification_with_worker(
                QualificationTypeId=self.counter_qual_id,
                WorkerId=worker_id,
                IntegerValue=incr,
                SendNotification=False
            )
        
        # if over limit, associate worker with limit achieved qualification
        if incr > self.counter_limit:
            print('Over limit')
            
            self.mturk.associate_qualification_with_worker(
                QualificationTypeId=self.limit_qual_id,
                WorkerId=worker_id,
                IntegerValue=1,
                SendNotification=False
            )

    def listener_bogus_qualification(self, sleep_normal=2, sleep_longer=3):
        """ This function should run separately to ensure qualifications are being assigned.

        :param handle_qualification: function to handle the response of a given worker to an assignment, extracting the
        worker id and the qualification name. This qualification will then be assigned to the worker.
        :param sleep_normal: How much time until next request to Amazon SQS in case there is a new assignment.
        :param sleep_longer: How much time until next request to Amazon SQS in case there is no new assignment.
        :return: Nothing.
        """

        while True:

            response = self.sqs.receive_message(QueueUrl=self.queue_url,
                                                AttributeNames=['All'],
                                                MaxNumberOfMessages=10,
                                                WaitTimeSeconds=0
                                                )
            if "Messages" not in response:
                time.sleep(sleep_longer)
                continue

            messages = response["Messages"]

            for message in response['Messages']:
                receipt = message["ReceiptHandle"]

                body = json.loads(message["Body"])

                for events in body['Events']:
                    if 'WorkerId' in events:
                        self.increment_user_qualification(events['WorkerId'])

                self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)
                
            time.sleep(sleep_normal)
            
            
# hit_ids = []

# for i in tqdm(range(len(hit_df))):

#     response = helper.mturk.create_hit(
#         MaxAssignments=1,
#         AutoApprovalDelayInSeconds=3600,
#         LifetimeInSeconds=3600,
#         AssignmentDurationInSeconds=3600,
#         Reward='0.01',
#         Title='Select the channel that appeals the most to younger/older audiences',
#         Keywords='categorize, youtube, channel, similarity, thumbnail, video, image',
#         Description='Given two youtube channels, pick the one that appeals the most to younger/older audiences',
#         QualificationRequirements=[
#           {
#             "QualificationTypeId": '000000000000000000L0',
#             "Comparator": 'GreaterThanOrEqualTo',
#             "IntegerValues": [95]
#           },
#     #     {
#     #         "QualificationTypeId": '00000000000000000071',
#     #         "Comparator": 'EqualTo',
#     #         "LocaleValues": [
#     #           {
#     #               "Country": "US",
#     #          }
#     #         ]
#     #       },
#           {
#             "QualificationTypeId": limit_qual_id,
#             "Comparator": 'DoesNotExist'
#           },

#         ],
#         HITLayoutId=hit_layout_id,
#         HITLayoutParameters=[{'Name':k, 'Value':str(v)} for k,v in hit_df.iloc[i].items() if k in ALLOWED_KEYS]
#     )

#     hit_ids.append(response['HIT']['HITId'])
#     notification = {'Destination': helper.queue_url, 'Transport': 'SQS',
#                     'Version': '2014-08-15', 'EventTypes': ['AssignmentSubmitted']}

#     vals = helper.mturk.update_notification_settings(HITTypeId=response['HIT']['HITTypeId'],
#                                             Notification=notification, Active=True)


## Get back results

# response = helper.mturk.list_assignments_for_hit(
#     HITId="3K1H3NEY8ZI9QPCZS4OISFXQVT5GDS",
#     MaxResults=100
# )