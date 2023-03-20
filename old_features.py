from typing import NamedTuple
from collections import defaultdict

from pathlib import Path
import numpy as np
from jinja2 import Template
from rich.progress import track
import json
import itertools
import math

import template
from fns.shapes import is_contiguous
from features import process_ctx

import bitutils

def describe_dot_old(i, dot_strings, dots, size_color):
    rounded_dots = (dots.round(2) * 100).astype(int)
    num_dots = dots.shape[0]
    # (x, y, size, color)
    return dot_desc_template.render(
        id = i+1,
        x = rounded_dots[i, 0],
        y = rounded_dots[i, 1],
        size = rounded_dots[i, 2],
        color = rounded_dots[i, 3],
    )

def describe_dot(i, dot_strings, dots, size_color):
    size_map = template.size_map5
    color_map = template.color_map5
    size, color = size_color[i]
    #return f"{dot_strings[i]} size {size_map[size]} and color {color_map[color]}"
    return f"{dot_strings[i]} {size_map[size]} size and {color_map[color]} color"

def describe_dot_pair(
    i, j, dot_strings, dots,
    short=False, group_attributes=False,
):
    # does not use quantized properties
    dot1 = dot_strings[i]
    dot2 = dot_strings[j]
    x1, y1, s1, c1 = dots[i]
    x2, y2, s2, c2 = dots[j]

    # TODO: i think the y values are negated, so this needs to be flipped
    #vert_comp = "above" if y1 > y2 else "below"
    vert_comp = "above" if y1 < y2 else "below"
    hor_comp = "right" if x1 > x2 else "left"
    size_comp = "bigger" if s1 > s2 else "smaller"
    #col_comp = "darker" if c1 > c2 else "lighter"
    col_comp = "darker" if c1 < c2 else "lighter"

    if group_attributes:
        return f"{dot1} {vert_comp} {hor_comp} {size_comp} {col_comp} {dot2}"

    if not short:
        vert_str = f"{dot1} is {vert_comp} {dot2}"
        hor_str = f"{dot1} is {hor_comp} of {dot2}"
        size_str = f"{dot1} is {size_comp} than {dot2}"
        col_str = f"{dot1} is {col_comp} than {dot2}"
        return ", ".join([vert_str, hor_str, size_str, col_str])
    else:
        vert_str = f"{dot1} {vert_comp} {dot2}"
        hor_str = f"{dot1} {hor_comp} {dot2}"
        size_str = f"{dot1} {size_comp} {dot2}"
        col_str = f"{dot1} {col_comp} {dot2}"
        return ", ".join([vert_str, hor_str, size_str, col_str])

def get_relations(
    i, j, dots, eps=0.05,
):
    # does not use quantized properties
    x1, y1, s1, c1 = dots[i]
    x2, y2, s2, c2 = dots[j]

    # return binary vector with relations
    # [above, below, left, right, bigger, smaller, darker, lighter]
    #above = y1 > y2 + eps # flipped
    #below = y1 < y2 - eps # flipped
    below = y1 > y2 + eps
    above = y1 < y2 - eps
    left  = x1 < x2 - eps
    right = x1 > x2 + eps
    bigger = s1 > s2 + eps
    smaller = s1 < s2 - eps
    #darker = c1 > c2 + eps # flipped?
    #lighter = c1 < c2 - eps # flipped?
    lighter = c1 > c2 + eps
    darker = c1 < c2 - eps

    return np.array([
        above, below,
        left, right,
        bigger, smaller,
        darker, lighter,
    ], dtype=bool)

def describe_relations(relation_vector):
    strings = [
        "above", "below",
        "left", "right",
        "bigger", "smaller",
        "darker", "lighter",
    ]
    assert len(strings) == len(relation_vector)
    return " ".join([s for s,b in zip(strings, relation_vector) if b])

def describe_dot_tgts(
    i, js, dot_strings, dots,
):
    """
    Describe dot1 {relation} dot2s
    """
    # do not use quantized properties
    dot1 = dot_strings[i]
    dot2s = [dot_strings[j] for j in js]

    x1, y1, s1, c1 = dots[i]
    x2s = dots[js, 0]
    y2s = dots[js, 1]
    s2s = dots[js, 2]
    c2s = dots[js, 3]

    # TODO: i think the y values are negated, so this needs to be flipped
    aboves = y1 > y2s
    belows = y1 < y2s
    rights = x1 > x2s
    lefts = x1 < x2s
    biggers = s1 > s2s
    smallers = s1 < s2s
    darkers = c1 > c2s
    lighters =  c1 < c2s

    comps = []
    if aboves.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if aboves[i]])
        comps.append(f"{dot1} above {sdots}")
    if belows.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if belows[i]])
        comps.append(f"{dot1} below {sdots}")
    if rights.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if rights[i]])
        comps.append(f"{dot1} right {sdots}")
    if lefts.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if lefts[i]])
        comps.append(f"{dot1} left {sdots}")
    if biggers.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if biggers[i]])
        comps.append(f"{dot1} bigger {sdots}")
    if smallers.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if smallers[i]])
        comps.append(f"{dot1} smaller {sdots}")
    if darkers.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if darkers[i]])
        comps.append(f"{dot1} darker {sdots}")
    if lighters.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if lighters[i]])
        comps.append(f"{dot1} lighter {sdots}")

    return ", ".join(comps)


def unit_vector(vector, axis=None):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector, axis=axis, keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_angles(xys):
    num_dots = xys.shape[0]
    pairs = [
        [tgt for tgt in range(num_dots) if src != tgt]
        for src in range(num_dots)
    ]
    xy_pairs = xys[np.array(pairs)]
    diffs = xys[:,None] - xy_pairs
    diffs = unit_vector(diffs, 1)
    return np.arccos(np.clip(
        (diffs[:,0] * diffs[:,1]).sum(-1),
        -1, 1
    ))

    # buggy?
    flat_diffs = diffs.reshape(-1, 2)
    return np.arccos(np.clip((
        flat_diffs[:,None] * flat_diffs
    ).sum(-1), -1., 1,))


def describe_dots(
    dots,
    use_short_describe = True,
    use_pairwise_features = True,
    use_unordered_pairwise = True,
    use_short_pairwise = True,
):
    dots = np.array(dots, dtype=float).reshape(-1, 4)
    dots[:,1] *= -1
    rounded_dots = (dots.round(2) * 100).astype(int)

    num_dots = dots.shape[0]
    dot_strings = [f"dot{i}" for i in range(1, num_dots+1)]

    """
    # (x, y, size, color)
    description_old = " [SEP] ".join([
        dot_desc_template.render(
            id = i+1,
            x = rounded_dots[i, 0],
            y = rounded_dots[i, 1],
            size = rounded_dots[i, 2],
            color = rounded_dots[i, 3],
        ) for i in range(num_dots)
    ])
    """

    ctx = process_ctx(dots)
    describe_dot_fn = describe_dot if use_short_describe else describe_dot_old

    descs = [describe_dot_fn(i, dot_strings, dots, ctx) for i in range(num_dots)]
    description = " [SEP] ".join(descs)

    if use_pairwise_features:
        # construct pairwise descriptions for each dot and 3 closest
        xy = dots[:,:2]
        dists = ((xy[:,None] - xy) ** 2).sum(-1)
        dists[range(7), range(7)] = dists.max() + 1
        closest_idxs = dists.argsort()[:,:2]

        # ordered pairs
        dot_pairs = [(i, j) for i in range(7) for j in closest_idxs[i]]
        if use_unordered_pairwise:
            # unordered pairs
            dot_pairs = set([tuple(sorted(x)) for x in dot_pairs])

        pairwise_strs = []
        for i,j in dot_pairs:
            pairwise_strs.append(describe_dot_pair(
                i, j, dot_strings, dots,
                short = use_short_pairwise,
            ))

        pairwise_str = ", ".join(pairwise_strs)
        description = f"{description} [SEP] {pairwise_str}"

    return description


def describe_plan_specific_dots(
    dots,
    plan,
    use_unordered_pairwise = True,
    close_dots = None,
    use_short_pairwise = True,
    use_config = True,
):
    extras = None

    boolplan = plan.astype(bool)
    dots = np.array(dots, dtype=float).reshape(-1, 4)
    rounded_dots = (dots.round(2) * 100).astype(int)

    num_dots = dots.shape[0]
    dot_strings = [f"dot{i}" for i in range(1, num_dots+1)]

    ctx = process_ctx(dots)
    descs = [
        describe_dot(i, dot_strings, dots, ctx)
        for i in range(num_dots)
        if boolplan[i]
    ]
    description = " [SEP] ".join(descs)

    if use_config:
        # only run this in plan-specific, since it can get really slow
        num_dots = dots.shape[0]
        config_sizes = [2,3]
        #config_sizes = [3]
        config_descs = []
        triangle_configs = []
        line_configs = []
        for size in config_sizes:
            plan_dots = plan.nonzero()[0]
            combinations = list(itertools.combinations(plan_dots, size))
            for idxs in combinations:
                # dots: (x,y,size,color)
                config = dots[idxs,:]
                xy = config[:,:2]
                pairwise_dists = ((xy[:,None] - xy) ** 2).sum(-1)

                # describe_config(config, size)
                if size == 2:
                    # TODO: fold this into pairwise
                    dist = pairwise_dists[0,1]
                    # hard-coded threshold
                    if dist < 0.1:
                        config_descs.append(f"dot{str(idxs[0]+1)} close dot{str(idxs[1]+1)}")
                elif size == 3:
                    multihot = np.zeros(7, dtype=bool)
                    multihot[list(idxs)] = True
                    #contig = is_contiguous(multihot, dots[:,:2], 7)
                    #contig = is_contiguous(dots[multihot], dots)
                    contig = is_contiguous(multihot, dots)
                    angles = get_angles(xy)
                    max_angle = angles.max() * 180 / math.pi
                    # hard-coded threshold
                    #if max_angle > 170 and contig:
                    if max_angle > 135:
                        config_descs.append(
                            f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                            "line"
                        )
                        line_configs.append(idxs)
                    elif max_angle <= 135 and contig:
                        config_descs.append(
                            f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                            "triangle"
                        )
                        triangle_configs.append(idxs)
        if len(config_descs) > 0:
            config_descriptions = " [SEP] ".join(config_descs)
            description = f"{description} [SEP] {config_descriptions}"
            """
            extras = GenerationExtras(
                triangle_configs = triangle_configs,
                line_configs = line_configs,
            )
            """

    # construct pairwise features for dot pairs in plan
    dot_pairs = [
        (i, j) for i in range(7) for j in range(7)
        if i != j and boolplan[i] and boolplan[j]
    ]
    if use_unordered_pairwise:
        # unordered pairs
        dot_pairs = set([tuple(sorted(x)) for x in dot_pairs])

    pairwise_strs = []
    for i,j in dot_pairs:
        pairwise_strs.append(describe_dot_pair(
            i, j,
            dot_strings,
            dots,
            short = use_short_pairwise,
            group_attributes = True,
        ))
        if close_dots is not None:
            raise NotImplementedError("Need to implement distance")

    pairwise_str = " , ".join(pairwise_strs)
    description = f"{description} [SEP] {pairwise_str}"

    return description, extras

def describe_mention(idxs, dots):
    size = len(idxs)
    config = dots[idxs,:]
    xy = config[:,:2]
    pairwise_dists = ((xy[:,None] - xy) ** 2).sum(-1)

    # describe all shared properties?

    # describe_config(config, size)
    if size == 2:
        # TODO: fold this into pairwise
        dist = pairwise_dists[0,1]
        # hard-coded threshold
        if dist < 0.1:
            return f"dot{str(idxs[0]+1)} close dot{str(idxs[1]+1)}"
        else:
            return describe_set(idxs)
    elif size == 3:
        multihot = np.zeros(7, dtype=bool)
        multihot[list(idxs)] = True
        #contig = is_contiguous(multihot, dots[:,:2], 7)
        #contig = is_contiguous(dots[multihot], dots)
        contig = is_contiguous(multihot, dots)
        angles = get_angles(xy)
        max_angle = angles.max() * 180 / math.pi
        # hard-coded threshold
        #if max_angle > 170 and contig:
        if max_angle > 135:
            return (
                f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                "line"
            )
        elif max_angle <= 135 and contig:
            return (
                f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                "triangle"
            )
        else:
            return describe_set(idxs)
    elif size == 4:
        multihot = np.zeros(7, dtype=bool)
        multihot[list(idxs)] = True
        contig = is_contiguous(multihot, dots[:,:2], 7)
        angles = get_angles(xy)
        max_angle = angles.max() * 180 / math.pi
        # hard-coded threshold
        #if max_angle > 170 and contig:
        if max_angle > 135:
            return (
                f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                "line"
            )
            # TODO: add more configs later
        else:
            return describe_set(idxs)
    else:
        return describe_set(idxs)

def describe_set(dots):
    return " ".join(f"dot{d+1}" for d in dots)

def describe_mention_specific_dots(
    dots,
    plan,
    mentions,
):
    extras = None

    boolplan = plan.astype(bool)
    boolmentions = mentions.astype(bool)
    dots = np.array(dots, dtype=float).reshape(-1, 4)
    rounded_dots = (dots.round(2) * 100).astype(int)

    num_dots = dots.shape[0]
    #dot_strings = [f"dot{i}" for i in range(1, num_dots+1)]
    dot_strings = [f"dot {i}" for i in range(1, num_dots+1)]

    ctx = process_ctx(dots)
    # unary
    descs = [
        describe_dot(i, dot_strings, dots, ctx)
        for i in range(num_dots)
        if boolplan[i]
    ]
    description = " [SEP] ".join(descs)

    # TODO: replace unary dot descriptions for mention descriptions
    mentionsets = [set(m.nonzero()[0]) for m in mentions]
    if len(mentionsets) == 0:
        # no mentions to describe
        return "none"

    mention_descriptions = [
        describe_mention(tuple(mentionsets[0]), dots)
    ]

    for src_mention, tgt_mention in zip(mentionsets, mentionsets[1:]):
        src_str = describe_set(src_mention)
        tgt_str = describe_set(tgt_mention)

        src_diff = src_mention.difference(tgt_mention)

        # get relations from src_diff dots to tgt
        relation_intersection = None
        for src in src_diff:
            for tgt in tgt_mention:
                relation_set = get_relations(src, tgt, dots)
                relation_intersection = (
                    relation_set
                    if relation_intersection is None
                    else relation_intersection & relation_set
                )
        if relation_intersection is None:
            #mention_descriptions.append(f"{src_str} none {tgt_str}")
            mention_descriptions.append(f"none")
        else:
            relation_string = describe_relations(relation_intersection)
            mention_descriptions.append(relation_string)
            #mention_descriptions.append(
                #f"{src_str} {relation_string} {tgt_str}"
            #)
        mention_descriptions.append(describe_mention(tuple(tgt_mention), dots))

    mention_description = " [SEP] ".join(mention_descriptions)

    return descs, mention_descriptions
    #return f"{description} [MSEP] {mention_description}"


if __name__ == "__main__":
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

    plan_idx = [0,1,4]
    plan = np.zeros(7, dtype=np.int8)
    plan[plan_idx] = 1

    mentions = plan[None]
    mentions = np.array([
        [1,1,0,0,0,0,0],
        [0,0,0,0,1,0,0],
    ])

    # mention_out looks pretty good
    mention_out = describe_mention_specific_dots(ctx, plan, mentions)
    dots_out = describe_dots(ctx)
    plan_out = describe_plan_specific_dots(ctx, plan)
    print(mention_out)
    print(dots_out)
    print(plan_out)

