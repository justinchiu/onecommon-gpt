from jinja2 import Template

RADIUS = .2
size_map3 = ["small", "medium", "large"]
color_map3 = ["dark", "grey", "light"]
#color_map3 = ["light", "grey", "dark"]

size_map5 = ["very small", "small", "medium", "large", "very large"]
# flipped?
color_map5 = ["very dark", "dark", "grey", "light", "very light"]
#color_map5 = ["very light", "light", "grey", "dark", "very dark"]

utterance_template = Template("{{ confirm }} {{ mention }}")

mention_wrapper = Template("Do you see any configurations that consist of {{mentions}} ?")

confirm = "Yes ."
disconfirm = "No ."

# template for individual dot features
dot_template = Template("{{size}}-sized and {{color}}")
named_dot_template = Template("{{size}}-sized and {{color}}{%if name is string%} ({{name}}){%endif%}")
spatial_dot_template = Template("{{spatial}} dot is {{dot}}")
named_spatial_dot_template = Template("{{spatial}} dot{%if name is string%} ({{name}}){%endif%} is {{dot}}")

# mention 1
# Do you see a small grey dot?
# mention 2
# Do you see a pair of dots, where the bottom left one is small and grey and the top right one is medium and light?
# mention 3
# Do you see three dots, where the {{dot 1}}, {{dot 2}}, and {{dot 3}}?

mention_1 = Template("a {{dot1}}")
mention_2 = Template("a pair of dots, where the {{dot1}} and the {{dot2}}")
mention_2a = Template("the {{dot1}} and the {{dot2}}")

mention_3 = Template("three dots, where the {{dot1}}, the {{dot2}}, and the {{dot3}}")

mention_41 = Template("four dots, where the {{dot1}}, the {{dot2}}, the {{dot3}}, and the {{dot4}}")
mention_42 = Template("four dots where three of them form a triangle, and the last dot falls in the triangle; the triangle has the {{dot1}}, the {{dot2}}, and the {{dot3}}")
mention_43 = Template("four dots where they roughly form a line; on one end it is {{dot1}}, and on the other end it is {{dot2}}; between the two dots, there are the {{dot3}} and the {{dot4}}.")

# selection
select_1 = Template("")

import numpy as np


class ConfigFeatures:
    def __init__(self, num_dots, sc, xy):
        self.num_dots = num_dots
        self.sc = sc
        self.xy = xy


    def set_dots(self):
        self.dots = [
            Dot(
                size = self.sc[i,0],
                color = self.sc[i, 1],
                xy = self.xy[i],
            ) for i in range(self.num_dots)
        ]


def is_only(dots, idx):
    return dots[idx] and dots.sum() == 1

def centroid(xy):
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    return (right + left) / 2, (top + bottom) / 2

def relative_position(x, y, mx, my, flip_y=True):
    # vertical stuff is flipped?
    if flip_y:
        if x < mx and y < my:
            return "top left"
        elif x > mx and y < my:
            return "top right"
        elif x < mx and y > my:
            return "bottom left"
        elif x > mx and y > my:
            return "bottom right"
        else:
            raise ValueError
    else:
        if x < mx and y < my:
            return "bottom left"
        elif x > mx and y < my:
            return "bottom right"
        elif x < mx and y > my:
            return "top left"
        elif x > mx and y > my:
            return "top right"
        else:
            raise ValueError

def spatial_descriptions2(xy, flip_y=True):
    assert xy.shape[0] == 2
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    mx, my = (right + left) / 2, (top + bottom) / 2

    horizontal_close = abs(right - left) < RADIUS
    vertical_close = abs(top - bottom) < RADIUS

    if horizontal_close and not vertical_close:
        # check if dots are close horizontally
        if flip_y:
            return [
                # looks like vertical stuff is flipped?
                #"top" if xy[0,1] > my else "bottom",
                #"top" if xy[1,1] > my else "bottom",
                "top" if xy[0,1] < my else "bottom",
                "top" if xy[1,1] < my else "bottom",
            ]
        else:
            # unflipped
            return [
                "top" if xy[0,1] > my else "bottom",
                "top" if xy[1,1] > my else "bottom",
                #"top" if xy[0,1] < my else "bottom",
                #"top" if xy[1,1] < my else "bottom",
            ]
    elif vertical_close and not horizontal_close:
        # check if dots are close vertically
        return [
            "left" if xy[0,0] < mx else "right",
            "left" if xy[1,0] < mx else "right",
        ]
    else:
        # otherwise use full description
        #rp0 = relative_position(xy[0,0], xy[0,1], mx, my)
        #rp1 = relative_position(xy[1,0], xy[1,1], mx, my)
        return [
            relative_position(xy[0,0], xy[0,1], mx, my, flip_y),
            relative_position(xy[1,0], xy[1,1], mx, my, flip_y),
        ]


def spatial_descriptions3(xy):
    assert xy.shape[0] == 3
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    mx, my = (right + left) / 2, (top + bottom) / 2

    # flipped
    #top_dots = xy[:,1] > my + RADIUS
    #bottom_dots = xy[:,1] < my - RADIUS
    bottom_dots = xy[:,1] > my + RADIUS
    top_dots = xy[:,1] < my - RADIUS

    right_dots = xy[:,0] > mx + RADIUS
    left_dots = xy[:,0] < mx - RADIUS

    # possible configurations:
    # * full rank triangle
    # * low rank line
    # * single dot? not likely
    descriptions = []
    for idx in range(xy.shape[0]):
        is_top = top_dots[idx]
        is_bottom = bottom_dots[idx]
        is_left = left_dots[idx]
        is_right = right_dots[idx]

        if is_only(top_dots, idx):
            descriptions.append("top")
        elif is_only(bottom_dots, idx):
            descriptions.append("bottom")
        elif is_only(left_dots, idx):
            descriptions.append("left")
        elif is_only(right_dots, idx):
            descriptions.append("right")
        elif is_top and is_left:
            descriptions.append("top left")
        elif is_top and is_right:
            descriptions.append("top right")
        elif is_bottom and is_left:
            descriptions.append("bottom left")
        elif is_bottom and is_right:
            descriptions.append("bottom right")
        else:
            raise ValueError
    return descriptions

def spatial_descriptions4(xy):
    assert xy.shape[0] == 4
    right, top = xy.max(0)
    left, bottom = xy.min(0)
    mx, my = (right + left) / 2, (top + bottom) / 2

    # flipped
    #top_dots = xy[:,1] > my + RADIUS
    #bottom_dots = xy[:,1] < my - RADIUS
    bottom_dots = xy[:,1] > my + RADIUS
    top_dots = xy[:,1] < my - RADIUS

    right_dots = xy[:,0] > mx + RADIUS
    left_dots = xy[:,0] < mx - RADIUS

    # possible configurations:
    # * full rank quadrilateral
    # * low rank triangle
    # * low rank line
    # * single dot
    descriptions = []
    for idx in range(xy.shape[0]):
        is_top = top_dots[idx]
        is_bottom = bottom_dots[idx]
        is_left = left_dots[idx]
        is_right = right_dots[idx]

        if is_only(top_dots, idx):
            descriptions.append("top")
        elif is_only(bottom_dots, idx):
            descriptions.append("bottom")
        elif is_only(left_dots, idx):
            descriptions.append("left")
        elif is_only(right_dots, idx):
            descriptions.append("right")
        elif is_top and is_left:
            descriptions.append("top left")
        elif is_top and is_right:
            descriptions.append("top right")
        elif is_bottom and is_left:
            descriptions.append("bottom left")
        elif is_bottom and is_right:
            descriptions.append("bottom right")
        else:
            raise ValueError
    return descriptions

def get_sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def check_triangle(p, p1, p2, p3):
    d1 = get_sign(p, p1, p2)
    d2 = get_sign(p, p3, p1)
    d3 = get_sign(p, p2, p3)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return (not (has_neg and has_pos))

def spatial_descriptions4(xy):
    assert xy.shape[0] == 4
    option = 1
    for i in range(4):
        compl = list(set([0, 1, 2, 3]) - set([i]))
        res = check_triangle(xy[i], xy[compl[0]], xy[compl[1]], xy[compl[2]])
        if res:
            option = 2
            dot4 = i
            descriptions = spatial_descriptions3(xy[compl, :])
            #descriptions.append("middle")
            descriptions.insert(i, "middle")
            break
    if option == 1:
        descriptions = []
        right, top = xy.max(0)
        left, bottom = xy.min(0)
        mx, my = (right + left) / 2, (top + bottom) / 2

        # flipped
        #top_dots = xy[:,1] > my + RADIUS
        #bottom_dots = xy[:,1] < my - RADIUS
        bottom_dots = xy[:,1] > my + RADIUS
        top_dots = xy[:,1] < my - RADIUS

        right_dots = xy[:,0] > mx
        left_dots = xy[:,0] < mx
        for idx in range(xy.shape[0]):
            is_top = top_dots[idx]
            is_bottom = bottom_dots[idx]
            is_left = left_dots[idx]
            is_right = right_dots[idx]

            if is_only(top_dots, idx):
                descriptions.append("top")
            elif is_only(bottom_dots, idx):
                descriptions.append("bottom")
            elif is_only(left_dots, idx):
                descriptions.append("left")
            elif is_only(right_dots, idx):
                descriptions.append("right")
            elif is_top and is_left:
                descriptions.append("top left")
            elif is_top and is_right:
                descriptions.append("top right")
            elif is_bottom and is_left:
                descriptions.append("bottom left")
            elif is_bottom and is_right:
                descriptions.append("bottom right")
            else:
                raise ValueError
    return descriptions


def size_color_descriptions(sc, size_map=size_map5, color_map=color_map5):
    #size_map = size_map3 if num_buckets == 3 else size_map5
    #color_map = color_map3 if num_buckets == 3 else color_map5
    return [
        (size_map[x[0]], color_map[x[1]]) for x in sc
    ]

def render_2(
    xy, sc,
    names=None, flip_y=True, concise=False,
    size_map=size_map5,
    color_map=color_map5,
):
    xy_desc = spatial_descriptions2(xy, flip_y)
    sc_desc = size_color_descriptions(sc, size_map, color_map)
    mention = mention_2a if concise else mention_2
    return mention.render(
        dot1 = named_spatial_dot_template.render(
            spatial = xy_desc[0],
            dot = dot_template.render(
                size = sc_desc[0][0],
                color = sc_desc[0][1],
            ),
            name = names[0] if names is not None else None,
        ),
        dot2 = named_spatial_dot_template.render(
            spatial = xy_desc[1],
            dot = dot_template.render(
                size = sc_desc[1][0],
                color = sc_desc[1][1],
            ),
            name = names[1] if names is not None else None,
        ),
    )

# function for rendering triangles, ignoring low rank cases
def render_3(xy, sc):
    xy_desc = spatial_descriptions3(xy)
    sc_desc = size_color_descriptions(sc)
    return mention_3.render(
        dot1 = spatial_dot_template.render(
            spatial = xy_desc[0],
            dot = dot_template.render(
                size = sc_desc[0][0],
                color = sc_desc[0][1],
            ),
        ),
        dot2 = spatial_dot_template.render(
            spatial = xy_desc[1],
            dot = dot_template.render(
                size = sc_desc[1][0],
                color = sc_desc[1][1],
            ),
        ),
        dot3 = spatial_dot_template.render(
            spatial = xy_desc[2],
            dot = dot_template.render(
                size = sc_desc[2][0],
                color = sc_desc[2][1],
            ),
        ),
    )

def render_4(xy, sc):
    xy_desc = spatial_descriptions4(xy)
    sc_desc = size_color_descriptions(sc)
    return mention_41.render(
        dot1 = spatial_dot_template.render(
            spatial = xy_desc[0],
            dot = dot_template.render(
                size = sc_desc[0][0],
                color = sc_desc[0][1],
            ),
        ),
        dot2 = spatial_dot_template.render(
            spatial = xy_desc[1],
            dot = dot_template.render(
                size = sc_desc[1][0],
                color = sc_desc[1][1],
            ),
        ),
        dot3 = spatial_dot_template.render(
            spatial = xy_desc[2],
            dot = dot_template.render(
                size = sc_desc[2][0],
                color = sc_desc[2][1],
            ),
        ),
        dot4 = spatial_dot_template.render(
            spatial = xy_desc[3],
            dot = dot_template.render(
                size = sc_desc[3][0],
                color = sc_desc[3][1],
            ),
        ),
    )

def render(n, sc, xy):
    if n == 1:
        return render_1(xy, sc)
    elif n == 2:
        return render_2(xy, sc)
    elif n == 3:
        return render_3(xy, sc)
    elif n == 4:
        return render_4(xy, sc)
    elif n == 5:
        return render_5(xy, sc)
    else:
        raise ValueError
 
if __name__ == "__main__":
    from belief import process_ctx, OrBelief

    num_dots = 7
    
    # scenario S_pGlR0nKz9pQ4ZWsw
    # streamlit run main.py
    ctx = np.array([
        0.635, -0.4,   2/3, -1/6,  # 8
        0.395, -0.7,   0.0,  3/4,  # 11
        -0.74,  0.09,  2/3, -2/3,  # 13
        -0.24, -0.63, -1/3, -1/6,  # 15
        0.15,  -0.58,  0.0,  0.24, # 40
        -0.295, 0.685, 0.0, -8/9,  # 50
        0.035, -0.79, -2/3,  0.56, # 77
    ], dtype=float).reshape(-1, 4)
    ids = np.array([8, 11, 13, 15, 40, 50, 77], dtype=int)

    # goes in init
    size_color = process_ctx(ctx)
    xy = ctx[:,:2]

    utt = np.array([1,0,1,1,0,0,0])
    utt = np.array([1,0,1,1,0,1,0])

    belief = OrBelief(num_dots, ctx, absolute=True)
    utt_feats = belief.get_feats(utt)
    matches = belief.resolve_utt(*utt_feats)

    # generate utterance for particular dot in triangle
    # num dots: int, size_color: n x 2, xy positions: n x 2
    n, sc, xy = utt_feats
    #xy = np.random.uniform(low=-1, high=1, size=(4,2))
    xy4_spatial_descriptions = spatial_descriptions4(xy[:4])
    xy3_spatial_descriptions = spatial_descriptions3(xy[:3])
    xy2_spatial_descriptions = spatial_descriptions2(xy[:2])

    # example call for rendering
    print(render_3(xy[:3], sc[:3]))
    print(render_4(xy[:4], sc[:4]))

    import matplotlib.pyplot as plt
    from matplotlib import cm
    # sc = {0,1,2}
    plt.scatter(
        xy[:,0], xy[:,1],
        s = (sc[:,0] + 1) * 10,
        c = cm.Greys((sc[:,1] + 1) * 85),
        #cmap = cm.Greys,
    )
    plt.show()

    import pdb; pdb.set_trace()
