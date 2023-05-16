import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity

from itertools import chain
from sklearn.neighbors import NearestNeighbors

"""
Very slightly modified from https://github.com/CSSLab/social-dimensions
"""


class DimenGenerator:
    def __init__(self, vectors):
        self.vectors = vectors

        self.name_mapping = {name.lower(): name for name in self.vectors.index}
        comm_names = np.array(self.vectors.index)

        logging.info("Finding nearest neighbors")
        neigh = NearestNeighbors(
            algorithm="ball_tree", n_neighbors=11, metric="euclidean"
        ).fit(self.vectors)
        all_neighs = neigh.kneighbors(self.vectors, return_distance=False)[:, 1:]

        logging.info("Creating potential pair indices")
        sorted_neighs = np.sort(all_neighs).ravel()
        indices = np.array(
            list(chain.from_iterable([[x] * 10 for x in range(all_neighs.shape[0])]))
        )

        logging.info("Creating potential directions dataframe")
        indices_to_calc = (indices, sorted_neighs)
        idx = list(zip(comm_names[indices_to_calc[0]], comm_names[indices_to_calc[1]]))
        direc = (
            self.vectors.iloc[indices_to_calc[1]].values
            - self.vectors.iloc[indices_to_calc[0]].values
        )

        self.directions_to_score = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(idx), data=direc
        )

    def generate_dimensions_from_seeds(self, seeds):
        return list(map(lambda x: self.generate_dimension_from_seeds([x]), seeds))

    def generate_dimension_from_seeds(self, seeds):
        seed_directions = (
            self.vectors.loc[map(lambda x: x[1], seeds)].values
            - self.vectors.loc[map(lambda x: x[0], seeds)].values
        )

        seed_similarities = np.dot(self.directions_to_score, seed_directions.T)
        seed_similarities = np.amax(seed_similarities, axis=1)

        directions = self.directions_to_score.iloc[
            np.flip(seed_similarities.T.argsort())
        ]

        # How many directions to take?
        num_directions = 10

        # make directions unique subreddits (subreddit can only occur once)
        ban_list = [s for sd in seeds for s in sd]
        i = -1  # to filter out seed pairs
        while (i < len(directions)) and (i < (num_directions + 1)):
            ban_list.extend(directions.index[i])

            l0 = directions.index.get_level_values(0)
            l1 = directions.index.get_level_values(1)
            directions = directions[
                (np.arange(0, len(directions)) <= i)
                | ((~l0.isin(ban_list)) & (~l1.isin(ban_list)))
            ]

            i += 1

        # Add seeds to the top
        index_seeds = (seeds + list(directions.index))[:num_directions]
        directions = np.concatenate(
            [pd.DataFrame(seed_directions, index=seeds), directions]
        )

        direction_group = pd.DataFrame(directions[0:num_directions], index=index_seeds)

        dimension = np.sum(direction_group, axis=0).values

        return {
            "note": "generated from seed pairs",
            "seed": seeds,
            "vector": dimension,
            "left_comms": list(map(lambda x: x[0], direction_group.index)),
            "right_comms": list(map(lambda x: x[1], direction_group.index)),
        }
