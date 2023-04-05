from ocdata import get_data
import bitutils
from collections import defaultdict
import numpy as np
import pandas as pd
import shapely

import seaborn as sns
import matplotlib.pyplot as plt

def get_dists(data):
    # get average distances of mentioned dots
    dists = defaultdict(list)
    df = defaultdict(list)
    for x in data:
        ctx = x["context"]
        all_mentions = x["all_referents"]
        print(len(all_mentions), len(x["dialogue"]))
        for t, mentions in enumerate(all_mentions):
            for mention in mentions:
                target = np.array(mention["target"], dtype=bool)
                cfg = bitutils.config_to_int(target).item()

                dist = np.linalg.norm(ctx[target,:2] - ctx[target,None,:2]).item()
                dists[target.sum().item()].append(dist)

                length = target.sum().item()

                xys = ctx[target,:2]
                mp = shapely.MultiPoint(xys)
                radius = shapely.minimum_bounding_radius(mp)

                if length > 1:
                    df["dist"].append(dist)
                    df["len"].append(length)
                    df["radius"].append(radius)

    """
    avg_dists = {k: np.mean(v) for k,v in dists.items()}
    q75_dists = {k: np.quantile(v, 0.75) for k,v in dists.items()}
    q90_dists = {k: np.quantile(v, 0.9) for k,v in dists.items()}
    print(avg_dists)
    print(q75_dists)
    print(q90_dists)
    """

    df = pd.DataFrame.from_dict(df)

    g = sns.FacetGrid(df, col="len")
    g.map(sns.histplot, "dist")
    g.savefig("figures/distance_histplots.png")

    g = sns.FacetGrid(df, col="len")
    g.map(sns.histplot, "radius")
    g.savefig("figures/radius_histplots.png")


def get_pairwise_dists(data):
    # get average distances of consecutive mentions
    dists = defaultdict(list)
    df = defaultdict(list)
    for x in data:
        ctx = x["context"]
        all_mentions = x["all_referents"]

        if len(all_mentions) != len(x["dialogue"]):
            all_mentions = [x for x in all_mentions if len(x) > 0]

        for t, mentions in enumerate(all_mentions):
            for m1, m2 in zip(mentions[:-1], mentions[1:]):
                tokens = x["dialogue"][t].split()
                span = tokens[
                    #max(0, m1["begin"] - 2)
                    #:min(len(tokens), m2["end"] + 2)
                    m1["begin"]:m2["end"]
                ]
                if not("near" in span or "close" in span or "next" in span):
                    continue
                if "not" in span:
                    continue


                target1 = np.array(m1["target"], dtype=bool)
                target2 = np.array(m2["target"], dtype=bool)
                target = target1 | target2
                cfg = bitutils.config_to_int(target).item()

                dist = np.linalg.norm(ctx[target,:2] - ctx[target,None,:2]).item()
                dists[target.sum().item()].append(dist)

                length = target.sum().item()

                xys = ctx[target,:2]
                mp = shapely.MultiPoint(xys)
                radius = shapely.minimum_bounding_radius(mp)

                print(" ".join(span), radius)
                if radius > 0.5:
                    continue

                if length > 1:
                    df["dist"].append(dist)
                    df["len"].append(length)
                    df["radius"].append(radius)

    """
    avg_dists = {k: np.mean(v) for k,v in dists.items()}
    q75_dists = {k: np.quantile(v, 0.75) for k,v in dists.items()}
    q90_dists = {k: np.quantile(v, 0.9) for k,v in dists.items()}
    print(avg_dists)
    print(q75_dists)
    print(q90_dists)
    """

    df = pd.DataFrame.from_dict(df)

    g = sns.FacetGrid(df, col="len")
    g.map(sns.histplot, "dist")
    g.savefig("figures/pair_distance_histplots.png")

    g = sns.FacetGrid(df, col="len")
    g.map(sns.histplot, "radius")
    g.savefig("figures/pair_radius_histplots.png")


if __name__ == "__main__":
    train_data, valid_data = get_data()

    #get_dists(valid_data)
    get_pairwise_dists(valid_data)
