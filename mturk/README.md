# Amazon Mechanical Turk experiments

We validate our embeddings by doing two experiments, using Amazon Mturk.

## Similarity experiment

The goal of this experiment is to make sure that two channels considered similar by our embedding are also considered similar by humans.

### Description

Given three youtube channels, represented by their video thumbnails, pick the odd one out. One example of comparison is shown below:

<p align="center">
<img src="https://user-images.githubusercontent.com/32189761/226681281-c0a03cec-7acc-4cc8-91ce-a895de4336ab.png">
</p>


Each comparison is created by :

- Sampling one youtube channel at random, call it A
- Getting the nearest youtube channel to A, call it B
- Getting the n-th (n in {600,300,150} depending on batch) nearest youtube channel to A, make sure that it is closer to A than to B, call it C

Then, we ask the workers to rate which one is the odd one out. If the workers rate C as the odd one out, this means that the embedding was correct: A and B are more similar, than they are similar to C.


## Bradley terry experiment

The goal of this experiment is to rate the "partisan" dimension, by computing its rank correlation with a ranking obtained from humans using a bradley-terry (plackett-luce) model.

### Description

Given two YouTube channels, pick the one which appeals to a more left-wing audience. (Or right-wing, randomly defined).

<p align="center">
<img src="https://user-images.githubusercontent.com/32189761/226681346-6d605869-dfe6-48b7-b966-057a338ac660.png">
</p>

We want to validate the “partisan” social dimension for each of our YouTube channels.

Our dataset is created by sampling channels, and stratifying on their partisan and partisan-ness dimension. Bins are created according to the thresholds seen in figure below (axes are standardized). We get 25 bins and choose a total of 100 channels for this experiment, meaning 4 channels sampled per bin.

<p align="center">
<img src="https://user-images.githubusercontent.com/32189761/226685275-2288c5d6-e9d4-4d6c-a251-056dccabcde9.png">
</p>

We then sample pairs of channels : each channel is featured in 20 pairs, and is never paired twice with the same channel. 

We get a total of 1000 channel pairs. For each pair, one worker will have to select which channel they think is the one that appeals the most to a left-wing / right-wing audience (randomized per user).

The goal of this experiment is to use the Bradley-Terry model to use the results from workers to get a ranking for each channel with respect to the partisan dimension. We then compare this ranking to the one obtained by our embedding partisan dimension, using Kendall’s Rank correlation.

High correlation means that our embedding is able to capture which channels have a more right-wing or left-wing audience. We stratify by partisan-ness to enforce having highly political as well as less political channels in our dataset.
