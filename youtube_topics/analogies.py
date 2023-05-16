import ipywidgets as widgets
import operator
import plotly.io as pio
import numpy as np
import plotly.express as px
import umap
import pandas as pd
import itertools

from functools import reduce
from sklearn.neighbors import NearestNeighbors

pio.renderers.default = "notebook"


class AnalogiesMetric:
    """Scoring class for computing subreddit analogies

    Uses analogies of different types : university -> city, team-> sport, city->sport
    """

    def __init__(
        self,
        filtered_subs=None,
        na_sports_path="north_america_sports.csv",
        uni_city_path="uni_to_city.csv",
    ):
        na_sports = pd.read_csv(
            na_sports_path, header=None, names=["city", "sport", "team"]
        )
        uni_city = pd.read_csv(uni_city_path)

        self.uni_city_subset = uni_city[["university", "city"]].rename(
            columns={"university": "A", "city": "B"}
        )
        self.team_sport_subset = na_sports[["team", "sport"]].rename(
            columns={"team": "A", "city": "B"}
        )

        unique_sports = sorted(na_sports["sport"].unique())
        self.per_sport_team_city = [
            na_sports.query("sport == @sport")[["team", "city"]]
            for sport in unique_sports
        ]
        self.tuple_dfs = [
            self.uni_city_subset,
            self.team_sport_subset,
        ] + self.per_sport_team_city

        self.labels = ["uni→city", "team→sport"] + [f"{x}→city" for x in unique_sports]

        self.filtered_subs = filtered_subs

    def check_analogies(self, subname_df, n_neighbors=10, return_all_results=False):
        """Computes analogy score

        Assuming pairs (A1,B1), (A2,B2) :
        One Analogy is considered "validated" if the B1-A1+A2 vector is one of the `n_neighbors` neighbours of the B2 vector

        if return_all_results, returns all dataset results, otherwise, aggregates the results and returns
        (total validated analogies, number of analogies)
        """

        subnames = set(subname_df.index)

        if self.filtered_subs is not None:
            subnames = subnames.intersection(self.filtered_subs)

        # subreddit idx to name
        subnames = sorted(subnames)
        subname_df = subname_df.loc[subnames]
        id_to_name = pd.Series(subname_df.index).to_dict()

        # fit nearest neighbours to our vectors
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        neigh.fit(subname_df.values)

        res_dfs = []

        tuple_dfs = self.tuple_dfs

        # iterate over all datasets
        for tuple_df in tuple_dfs:
            tuple_df = tuple_df[
                reduce(
                    operator.and_,
                    (tuple_df[col].isin(subnames) for col in tuple_df.columns),
                )
            ]

            # compute all pairs
            df = pd.DataFrame(
                itertools.permutations(tuple_df.itertuples(index=False), 2)
            )

            if len(df) == 0:
                continue

            # create df with pairs as columns, make sure that b1 != b2
            df = pd.DataFrame.from_records(
                df[0] + df[1], columns=["A1", "B1", "A2", "B2"]
            )
            df = df[df["B1"] != df["B2"]].reset_index(drop=True)

            # compute distance vectors
            all_vecs = (
                subname_df.loc[df["B1"]].values
                - subname_df.loc[df["A1"]].values
                + subname_df.loc[df["A2"]].values
            )

            if len(all_vecs) == 0:
                continue

            # get neighbours of distance vectors
            df["top10"] = (
                pd.DataFrame(neigh.kneighbors(all_vecs, return_distance=False))
                .applymap(id_to_name.get)
                .apply(list, axis=1)
            )
            df["in10"] = df.apply(lambda x: x["B2"] in x["top10"], axis=1)

            res_dfs.append(df)

        # return all dataset results
        if return_all_results:
            return [
                (label, df["in10"].sum(), len(df))
                for label, df in zip(self.labels, res_dfs)
            ]

        # return total true, len
        else:
            res_df = pd.concat(res_dfs)

            if len(res_df) == 0:
                return 0, 0

            return res_df["in10"].sum(), len(res_df)


class SimilarWidget:
    """Widget that shows nearest neighbors of one vector in UMAP"""

    def __init__(self, vectors_df):
        """Create SimilarWidget

        Args:
            vectors_df: A dataframe of vectors, where the index is used as vector names
        """

        self.vectors_df = vectors_df

        # fit nearest neighbours
        neigh = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.neigh = neigh.fit(vectors_df.values)

        # fit umap embed
        self.fit = umap.UMAP(n_jobs=10)
        u = self.fit.fit_transform(vectors_df)

        # created df from embed
        self.embed_df = pd.DataFrame(u, index=vectors_df.index, columns=["x", "y"])

        # input text to select one vector (subreddit / channel) from its name
        self.a = widgets.Combobox(
            placeholder="Choose Someone",
            options=tuple(vectors_df.index),
            description="A:",
            ensure_option=True,
            disabled=False,
        )

    def display(self):
        """Display ipywidget with selector for vector

        Upon selection of a vector, shows its nearest neighbours projected using UMAP embedding
        """

        def f(a):
            if a != "":
                A = self.vectors_df.loc[a]
                idx = self.neigh.kneighbors(
                    A.values.reshape((1, -1)), return_distance=False
                )
                neighs = self.vectors_df.iloc[np.ravel(idx)]
                display(neighs)

                self.embed_df["color"] = "Neighbour"
                self.embed_df.loc[a, "color"] = "A"

                fig = px.scatter(
                    self.embed_df.loc[list(neighs.index) + [a]]
                    .reset_index()
                    .rename(columns={"index": "channelTitle"}),
                    x="x",
                    y="y",
                    hover_data=["channelTitle"],
                    color="color",
                    color_discrete_sequence=px.colors.qualitative.G10,
                )

                fig.show("notebook")

        widgets.interact(f, a=self.a)


class AnalogyWidget:
    """Widget that plots analogies in a UMAP embed"""

    def __init__(self, vectors_df, known_d=False):
        """Create SimilarWidget

        Args:
            vectors_df: A dataframe of vectors, where the index is used as vector names
            known_d: Whether to also ask for a fourth d vector, if we know what the result of the analogy should be
        """

        self.vectors_df = vectors_df

        neigh = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.neigh = neigh.fit(vectors_df.values)

        self.fit = umap.UMAP(n_neighbors=100, n_jobs=10, metric="cosine")
        u = self.fit.fit_transform(vectors_df)

        self.embed_df = pd.DataFrame(u, index=vectors_df.index, columns=["x", "y"])

        self.a = widgets.Combobox(
            placeholder="Choose Someone",
            options=tuple(vectors_df.index),
            description="A:",
            ensure_option=True,
            disabled=False,
        )

        self.b = widgets.Combobox(
            placeholder="Choose Someone",
            options=tuple(vectors_df.index),
            description="B:",
            ensure_option=True,
            disabled=False,
        )

        self.c = widgets.Combobox(
            placeholder="Choose Someone",
            options=tuple(vectors_df.index),
            description="C:",
            ensure_option=True,
            disabled=False,
        )

        if known_d:
            self.known_d = known_d
            self.d = widgets.Combobox(
                placeholder="Choose Someone",
                options=tuple(vectors_df.index),
                description="D:",
                ensure_option=True,
                disabled=False,
            )
        else:
            self.d = None

    def display(self):
        """Display ipywidget with selector for vectors

        Upon selection of all vectors (3 by default, 4 if d is not None),
        shows the vector obtained by b-a+c, and nearest neighbours of each vector
        """

        def f(a, b, c, d=None):
            if not any([a == "", b == "", c == ""]):
                ((_, A), (_, B), (_, C)) = list(
                    self.vectors_df.loc[[a, b, c]].iterrows()
                )
                idx = self.neigh.kneighbors(
                    (A - B + C).values.reshape((1, -1)), return_distance=False
                )
                display(self.vectors_df.iloc[np.ravel(idx)])

                get_neighs = lambda vec: list(
                    self.vectors_df.iloc[
                        np.ravel(
                            self.neigh.kneighbors(
                                vec.values.reshape((1, -1)), return_distance=False
                            )
                        )
                    ].index
                )

                a_neighs, b_neighs, c_neighs, d_neighs = (
                    get_neighs(A),
                    get_neighs(B),
                    get_neighs(C),
                    get_neighs(A - B + C),
                )

                all_neighs = [a_neighs, b_neighs, c_neighs, d_neighs] + (
                    [[d]] if d is not None else []
                )
                all_points = sorted(set(reduce(operator.add, all_neighs)))

                self.embed_df["color"] = "Neighbour"
                self.embed_df.loc[(self.embed_df.index == a), "color"] = "A"
                self.embed_df.loc[(self.embed_df.index == b), "color"] = "B"
                self.embed_df.loc[(self.embed_df.index == c), "color"] = "C"
                self.embed_df.loc[
                    (self.embed_df.index == d), "color"
                ] = "Expected Result"

                self.embed_df.loc[
                    (
                        self.embed_df.index.isin(d_neighs)
                        & (self.embed_df.index != d)
                        & (self.embed_df.index != a)
                        & (self.embed_df.index != b)
                        & (self.embed_df.index != c)
                    ),
                    "color",
                ] = "D results"

                ax, ay = self.embed_df.loc[a][["x", "y"]]
                bx, by = self.embed_df.loc[b][["x", "y"]]

                cx, cy = self.embed_df.loc[c][["x", "y"]]
                dx, dy = np.ravel(
                    self.fit.transform((A - B + C).values.reshape((1, -1)))
                )

                fig = px.scatter(
                    self.embed_df.loc[all_points]
                    .reset_index()
                    .rename(columns={"index": "channelTitle"}),
                    x="x",
                    y="y",
                    hover_data=["channelTitle"],
                    color="color",
                    color_discrete_sequence=px.colors.qualitative.G10,
                )

                fig.add_annotation(
                    ax=bx,
                    axref="x",
                    ay=by,
                    ayref="y",
                    x=ax,
                    arrowcolor="red",
                    xref="x",
                    y=ay,
                    yref="y",
                    arrowwidth=2.5,
                    arrowside="end",
                    arrowsize=1,
                    arrowhead=4,
                )

                fig.add_annotation(
                    ax=cx,
                    axref="x",
                    ay=cy,
                    ayref="y",
                    x=dx,
                    arrowcolor="red",
                    xref="x",
                    y=dy,
                    yref="y",
                    arrowwidth=2.5,
                    arrowside="end",
                    arrowsize=1,
                    arrowhead=4,
                )

                fig.show("notebook")

        widgets.interact(f, a=self.a, b=self.b, c=self.c, d=self.d)
