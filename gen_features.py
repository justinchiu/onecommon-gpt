# Features for generation

from collections import defaultdict
from itertools import combinations
import numpy as np

from fns.shapes import is_triangle, is_line, is_square, is_contiguous
from fns.size import is_large, is_small, all_size
from fns.color import all_color
from fns.spatial import (
    all_close, are_close,
    are_above, are_below, are_right, are_left,
    get_top, get_bottom, get_right, get_left,
    get_top_right, get_top_left, get_bottom_right, get_bottom_left,
)

from features import process_ctx, size_color_descriptions
from old_features import describe_mention_specific_dots

unary_functions = {
    "all_close": all_close,
    #"all_color": all_color,
    #"all_size": all_size,
}

binary_functions = {
    "is_triangle": is_triangle,
    "is_line": is_line,
    "is_square": is_square,
    "is_contiguous": is_contiguous,
}

def get_features(ctx):
    xy = ctx[:,:2]
    sc = process_ctx(ctx)

    set_features = {None: {}}
    costs = defaultdict(int)
    for n in range(1,5):
        for idxs in combinations(range(7), n):
            ndots = len(idxs)
            idxs = list(idxs)
            dots = ctx[idxs]
            xys = xy[idxs]
            scs = sc[idxs]
            sizes = scs[:,0]
            colors = scs[:,1]

            counts = {
                "colors": len(set(colors)),
                "sizes": len(set(sizes)),
            }

            unarys = {
                k: fn(idxs, ctx)
                for k, fn in unary_functions.items()
            }
            binarys = {
                k: fn(idxs, ctx)
                for k, fn in binary_functions.items()
            }
            set_features[tuple(idxs)] = counts | unarys | binarys

            color_size_cost = sum(counts.values())
            dist_cost = 1 if unarys["all_close"] else ndots
            shape_cost = 1 if any(binarys.values()) else ndots

            costs[tuple(idxs)] = color_size_cost + dist_cost + shape_cost
    return set_features, costs

# only into two partitions
def partitions(plan):
    total_bits = 8
    size = plan.sum().item()
    num_configs = 2 ** size
    configs = np.arange(num_configs, dtype=np.uint8)

    masks = np.unpackbits(configs[:,None], axis=-1)[:,-size:].astype(bool)
    num_masks = len(masks)
    complements = ~masks
    dot_idxs = plan.nonzero()[0]

    idxs = [
        (
            tuple(dot_idxs[mask]) if mask.sum() > 0 else None,
            tuple(dot_idxs[complement]) if complement.sum() > 0 else None,
        )
        for mask, complement in zip(masks[:num_masks//2], complements[:num_masks//2])
    ]
    return idxs

def get_mention(parts):
    filtered_parts = [x for x in parts if x is not None]
    num_parts = len(filtered_parts)
    mentions = np.zeros((num_parts, 7))
    for part in filtered_parts:
        mentions[part] = 1
    return mentions


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    ctx = np.array([
        0.635, -0.4,   2/3, -1/6,  # 8
        0.395, -0.7,   0.0,  3/4,  # 11
        -0.74,  0.09,  2/3, -2/3,  # 13
        -0.24, -0.63, -1/3, -1/6,  # 15
        0.15,  -0.58,  0.0,  0.24, # 40
        -0.295, 0.685, 0.0, -8/9,  # 50
        0.035, -0.79, -2/3,  0.56, # 77
    ], dtype=float).reshape(-1, 4)
    real_ids = np.array(['8', '11', '13', '15', '40', '50', '77'], dtype=int)
    # reflect across y axis
    ctx[:,1] = -ctx[:,1]
    #xy = np.random.rand((7,2)) * 2 - 1
    xy = ctx[:,:2]
    sc = process_ctx(ctx)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(
        xy[:,0], xy[:,1],
        marker='o',
        s = 100 * (ctx[:,2] + 1),
        c = -ctx[:,3],
        cmap="binary",
        edgecolor="black",
        linewidth=1,
    )

    #plan = np.array([1,0,1,1,0,0,0], dtype=bool)
    plan = np.array([1,1,0,0,1,0,0], dtype=bool)
    ax.scatter(xy[plan,0], xy[plan,1], marker="x", s=100, c="r")
    for i, id in enumerate(real_ids):
        ax.annotate(id, (xy[i,0]+.025, xy[i,1]+.025))
    fig.savefig("view.png")
    """
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width=300)
    """

    features, costs = get_features(ctx)

    min_score = 100
    best_feats = None
    best_parts = None
    for parts in partitions(plan):
        parts = [part for part in parts if part]
        feats = [features[part] for part in parts]
        scores = [costs[part] for part in parts]
        transition_score = 0
        if len(parts) > 1:
            for cur, next in zip(parts[:-1], parts[1:]):
                import pdb; pdb.set_trace()
        score = sum(scores)
        print(parts)
        print(feats)
        print(scores)
        if score < min_score:
            min_score = score
            best_parts = parts
            best_feats = feats
    mentions = get_mention(parts)
    mention_desc = describe_mention_specific_dots(ctx, plan, mentions)
    print(min_score)
    print(best_feats)
    print(best_parts)
    print(mention_desc)
    import pdb; pdb.set_trace()

