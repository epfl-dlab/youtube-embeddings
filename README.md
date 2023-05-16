# Youtube topics

The goal of this repository is to create embeddings for YouTube channels.
These embeddings can be used as-is for content similarity, and can also be used to extract social dimensions.

### Types of Embeddings

We propose three types of embeddings.

- Social Sharing / Reddit embedding: made using shares of YouTube videos on Reddit (using [Pushshift](https://pushshift.io/) data)
- Content embedding: made from video titles and descriptions, fed through a Sentence Transformer.
- Recommendation embedding: made from recording recommendations YouTube provides to a history-less user, and computing a node embedding.

Those embeddings for our filtered 40K channels are featured in the `embeds/` folder.

Similarly, social dimensions are featured in the `dims/` folder.


### Recreating embeddings

Create a conda (/mamba) environment using `conda env create -f environment.yml`.
This creates a conda environment named `ytb` with all libraries necessary for running the code. It might be necessary to upgrade your conda version beforehand (`conda upgrade conda`) if you get any error.

The repository uses jupytext for notebooks version control, so notebooks are saved in Markdown format, which still makes them readable from github, and removes the output.

All of the notebooks for recreating the embeddings are in the `generate_embeddings/` folder. Please note that it will require some work to get everything working. Notably, it assumes you have already extracted all links to youtube in reddit comments and submissions (the pyspark code for extracting them is not (not yet?) public).

---
Unfortunately, it looks like the pushshift dumps are currently not accessible over on [https://files.pushshift.io/](https://files.pushshift.io/) (although there seems to be a torrent remaining), and according to [this post](https://www.reddit.com/r/modnews/comments/134tjpe/reddit_data_api_update_changes_to_pushshift_access/), Reddit revoked pushshift's access, so more recent posts will not be able to be included in datasets.

### Notebook & methods diagram


<picture style="background-color:none;visibility:visible;">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/boesingerl/youtube-embeds/assets/32189761/30a57202-f838-4098-bdb4-4136b298b4d9">
  <img alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="https://github.com/boesingerl/youtube-embeds/assets/32189761/bdb9989e-b60e-458a-a317-e92a0336faef">
</picture>
