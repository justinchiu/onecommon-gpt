from jinja2 import Template
import numpy as np

import template_rec

RADIUS = .2
size_map3 = ["small", "medium", "large"]
color_map3 = ["dark", "grey", "light"]
#color_map3 = ["light", "grey", "dark"]

size_map5 = ["very small", "small", "medium", "large", "very large"]
# flipped?
color_map5 = ["very dark", "dark", "grey", "light", "very light"]
#color_map5 = ["very light", "light", "grey", "dark", "very dark"]

def process_ctx(
    ctx,
    absolute=True,
    num_size_buckets = 5,
    num_color_buckets = 5,
):
    # ctx: [x, y, size, color]
    eps = 1e-3
    min_ = ctx.min(0)
    max_ = ctx.max(0)

    if absolute:
        # ABSOLUTE BUCKET
        min_size, max_size = -1, 1
        min_color, max_color = -1, 1
    else:
        # RELATIVE BUCKET
        min_size, max_size = min_[2], max_[2]
        min_color, max_color = min_[3], max_[3]
    

    size_buckets = np.linspace(min_size, max_size + eps, num_size_buckets+1)
    color_buckets = np.linspace(min_color, max_color + eps, num_color_buckets+1)
    sizes = ctx[:,2]
    colors = ctx[:,3]

    size_idxs = (size_buckets[:-1,None] <= sizes) & (sizes < size_buckets[1:,None])
    color_idxs = (color_buckets[:-1,None] <= colors) & (colors < color_buckets[1:,None])
    return np.stack((size_idxs.T.nonzero()[1], color_idxs.T.nonzero()[1]), 1)


def size_color_descriptions(sc, size_map=size_map5, color_map=color_map5):
    #size_map = size_map3 if num_buckets == 3 else size_map5
    #color_map = color_map3 if num_buckets == 3 else color_map5
    return [
        (size_map[x[0]], color_map[x[1]]) for x in sc
    ]


def get_feats(plan, xy, size_color):
    plan_size = plan.sum().item()
    plan_sc = size_color[plan.astype(bool)]
    plan_xy = xy[plan.astype(bool)]
    return plan_size, plan_sc, plan_xy


def render(plan, context, confirm=None):
    #plan = np.array([0,1,0,0,0,0,1])
    xy = context[:,:2]
    sc = process_ctx(context)
    feats = get_feats(plan, xy, sc)
    ids = plan.nonzero()[0]
    desc = template_rec.render(*feats, ids, confirm=confirm)
    return desc
