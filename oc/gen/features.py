from jinja2 import Template
import numpy as np

import oc.gen.template_rec as template_rec

from oc.fns.shapes import is_triangle, is_line, is_square
from oc.fns.spatial import all_close, is_above, is_below, is_right, is_left, is_middle
from oc.fns.spatial import get_top, get_bottom, get_right, get_left
from oc.fns.spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from oc.fns.spatial import get_middle
from oc.fns.spatial import get_distance, get_minimum_radius
from oc.fns.color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, is_darker, is_lighter
from oc.fns.size import is_large, is_small, is_medium_size, largest, smallest, same_size, different_size, is_larger, is_smaller
from oc.fns.iterators import get1idxs, get2idxs, get3idxs, getsets
from oc.fns.lists import add

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


def render(plan, context, confirm=None, num_buckets=3):
    #plan = np.array([0,1,0,0,0,0,1])
    xy = context[:,:2]
    sc = process_ctx(context, num_size_buckets=num_buckets, num_color_buckets=num_buckets)
    feats = get_feats(plan, xy, sc)
    ids = plan.nonzero()[0]
    desc = template_rec.render(*feats, ids, confirm=confirm, num_buckets=num_buckets)
    return desc

def new_vs_old_desc(newdot, olddots, ctx, num_buckets=3):
    right = all(is_right(newdot, dot, ctx) for dot in olddots)
    left = all(is_left(newdot, dot, ctx) for dot in olddots)
    above = all(is_above(newdot, dot, ctx) for dot in olddots)
    below = all(is_below(newdot, dot, ctx) for dot in olddots)
    middle = is_middle(newdot, olddots, ctx)

    if right and above:
        position_desc = "to the right and above"
    elif right and below:
        position_desc = "to the right and below"
    elif right:
        position_desc = "right of"
    elif left and above:
        position_desc = "to the left and above"
    elif left and below:
        position_desc = "to the left and below"
    elif left:
        position_desc = "left of"
    elif above:
        position_desc = "above"
    elif below:
        position_desc = "below"
    elif middle:
        position_desc = "in the middle of"
    else:
        position_desc = "near any of"
        #raise ValueError

    size_color = process_ctx(
        ctx,
        num_size_buckets=num_buckets,
        num_color_buckets=num_buckets,
    )
    dots2 = size_color[[newdot]]
    descs = size_color_descriptions(dots2, size_map=size_map3, color_map=color_map3)

    sc_old = size_color[olddots]
    old_descs = size_color_descriptions(sc_old, size_map=size_map3, color_map=color_map3)

    return descs, position_desc, old_descs
