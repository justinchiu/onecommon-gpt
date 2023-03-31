from ocdata import get_data
import bitutils
from collections import defaultdict
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def get_dists(data):
    # get average distances of mentioned dots
    dists = defaultdict(list)
    df = defaultdict(list)
    for x in data:
        ctx = x["context"]
        all_mentions = x["all_referents"]
        for mentions in all_mentions:
            for mention in mentions:
                target = np.array(mention["target"], dtype=bool)
                cfg = bitutils.config_to_int(target).item()
                dist = np.linalg.norm(ctx[target,:2] - ctx[target,None,:2]).item()
                dists[target.sum().item()].append(dist)
                length = target.sum().item()
                if length > 1:
                    df["dist"].append(dist)
                    df["len"].append(length)

    avg_dists = {k: np.mean(v) for k,v in dists.items()}
    q75_dists = {k: np.quantile(v, 0.75) for k,v in dists.items()}
    q90_dists = {k: np.quantile(v, 0.9) for k,v in dists.items()}
    print(avg_dists)
    print(q75_dists)
    print(q90_dists)

    df = pd.DataFrame.from_dict(df)

    g = sns.FacetGrid(df, col="len")
    g.map(sns.histplot, "dist")
    g.savefig("figures/distance_histplots.png")


if __name__ == "__main__":
    train_data, valid_data = get_data()

    get_dists(valid_data)
