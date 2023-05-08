import os
import openai
import json
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")

codes = [
"""
# Them: Got a triangle of 3 light grey dots by itself.
def turn(state):
    # New question.
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 3):
        for x,y,z in permutations(config):
            check_xyz_triangle = is_triangle([x,y,z], ctx)
            check_xyz_light = all([is_light(dot, ctx) for dot in [x,y,z]])
            check_xyz_alone = all([not all_close([x,y,z,dot], ctx) for dot in idxs if dot not in [x,y,z]])
            if (
                check_xyz_triangle
                and check_xyz_light
                and check_xyz_alone
            ):
                dots = frozenset([x,y,z])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.
""",
    """# You: Could be. One on right is largest with a tiny gray on top??
def turn(state):
    # Follow up question.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c in permutations(config):
            check_a_right = a == get_right([a,b,c], ctx)
            check_a_largest = a == largest([a,b,c], ctx)
            check_b_tiny = is_small(b, ctx)
            check_b_grey = is_grey(b, ctx)
            check_b_top = b == get_top([a,b,c], ctx)
            if (
                check_a_right
                and check_a_largest
                and check_b_tiny
                and check_b_grey
                and check_b_top
            ):
                dots = frozenset([a,b,c])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# Them: Nevermind. Do you see a pair of dark dots? One with another above and to the right of it? Same size as well.
def turn(state):
    # New question.
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 2):
        for x, y in permutations(config):
            check_xy_pair = all_close([x,y], ctx)
            check_xy_dark = is_dark(x, ctx) and is_dark(y, ctx)
            check_y_right_x = is_right(y, x, ctx)
            check_y_above_x = is_above(y, x, ctx)
            check_xy_same_size = same_size([x,y], ctx)
            if (
                check_xy_pair
                and check_xy_dark
                and check_y_right_x
                and check_y_above_x
                and check_xy_same_size
            ):
                dots = frozenset([x,y])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# You: No.
def turn(state):
    # New question.
    return []
state = turn(state)
# End.""",
    """# Them: What about a large medium grey dot near the center?
def turn(state):
    # New question.
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 1):
        for x, in permutations(config):
            check_x_large = is_large(x, ctx)
            check_x_grey = is_grey(x, ctx)
            check_x_center = is_middle(x, None, ctx)
            if (
                check_x_large
                and check_x_grey
                and check_x_center
            ):
                dots = frozenset([x])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# You: Is there a smaller black one next to it?
def turn(state):
    # Follow up question, new dot.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a, in permutations(config):
            for x, in get1idxs(idxs, exclude=[a]):
                check_x_smaller_a = is_smaller(x, a, ctx)
                check_x_dark = is_dark(x, ctx)
                check_x_next_to_a = all_close([a,x], ctx)
                if(
                    check_x_smaller_a
                    and check_x_dark
                    and check_x_next_to_a
                ):
                    dots = frozenset([a,x])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# Them: No. Do you see three dots in a diagonal line, where the top left dot is light, middle dot is grey, and bottom right dot is dark?
def turn(state):
    # New question.
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 3):
        for x,y,z in permutations(config):
            check_xyz_line = is_line([x,y,z], ctx)
            check_x_top_left = x == get_top_left([x, y, z], ctx)
            check_x_light = is_light(x, ctx)
            check_y_middle = is_middle(y, [x,y,z], ctx)
            check_y_grey = is_grey(y, ctx)
            check_z_bottom_right = z == get_bottom_right([x, y, z], ctx)
            check_z_dark = is_dark(z, ctx)
            if (
                check_xyz_line
                and check_x_top_left
                and check_x_light
                and check_y_middle
                and check_y_grey
                and check_z_bottom_right
                and check_z_dark
            ):
                dots = frozenset([x,y,z])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# You: Yes. Is the top one close to the middle darker one?
def turn(state):
    # Follow up question.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c in permutations(config):
            check_a_top = a == get_top([a,b,c], ctx)
            check_b_middle = b == get_middle([a,b,c], ctx)
            check_ab_close = all_close([a, b], ctx)
            check_b_darker_a = is_darker(b, a, ctx)
            if (
                check_a_top
                and check_b_middle
                and check_ab_close
                and check_b_darker_a
            ):
                results.add(frozenset([a,b,c]))
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# Them: Yes. And the smallest is on the bottom right.
def turn(state):
    # Follow up question.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c in permutations(config):
            check_a_smallest = a == smallest([a,b,c], ctx)
            check_a_bottom_right = a == get_bottom_right([a,b,c], ctx)
            if (
                check_a_smallest
                and check_a_bottom_right
            ):
                dots = frozenset([a,b,c])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# You: Yes, let's select the large one. <selection>.
def select(state):
    # Select a dot.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c in permutations(config):
            check_a_large = is_large(a, ctx)
            check_b_not_large = not is_large(b, ctx)
            check_c_not_large = not is_large(c, ctx)
            if (
                check_a_large
                and check_b_not_large
                and check_c_not_large
            ):
                dots = frozenset([a])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=True)
state = select(state)
# End.""",
    """# You: Do you see a large black dot on the bottom left?
def turn(state):
    # New question.
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 1):
        for x, in permutations(config):
            check_x_large = is_large(x, ctx)
            check_x_dark = is_dark(x, ctx)
            check_x_below_left = is_below(x, None, ctx) and is_left(x, None, ctx)
            if (
                check_x_large
                and check_x_dark
                and check_x_below_left
            ):
                dots = frozenset([x])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# Them: I see a large black dot next to two smaller lighter dots. The two smaller ones are the same size and color. We have different views though.
def turn(state):
    # New question.
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 3):
        for x,y,z in permutations(config):
            check_xyz_close = all_close([x,y,z], ctx)
            check_x_large = is_large(x, ctx)
            check_z_dark = is_dark(z, ctx)
            check_y_smaller_x = is_smaller(y, x, ctx)
            check_z_smaller_x = is_smaller(z, x, ctx)
            check_y_lighter_x = is_lighter(y, x, ctx)
            check_z_lighter_x = is_lighter(z, x, ctx)
            check_yz_same_size = same_size([y,z], ctx)
            check_yz_same_color = same_color([y,z], ctx)
            if (
                check_xyz_close
                and check_x_large
                and check_z_dark
                and check_y_smaller_x
                and check_z_smaller_x
                and check_y_lighter_x
                and check_z_lighter_x
                and check_yz_same_size
                and check_yz_same_color
            ):
                dots = frozenset([x,y,z])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.""",
    """# You: Select the largest one.
def select(state):
    # Select a dot.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c in permutations(config):
            check_a_largest = a == get_largest([a,b,c], ctx)
            if (
                check_a_largest
            ):
                dots = frozenset([a])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=True)
state = select(state)
# End.""",
    """# Them: Okay.
def noop(state):
    # No op.
    return state
state = noop(state)
# End.""",
    """# You: Okay. <selection>.
def noop(state):
    # No op.
    return state
state = noop(state)
# End.""",
]

def construct_prompt(code):
 return f"""# Them: Got a triangle of 3 light grey dots by itself.
def turn(state):
    # New question.
    results = set()
    orderedresults = []
    parents = []
    for config in getsets(idxs, 3):
        for x,y,z in permutations(config):
            check_xyz_triangle = is_triangle([x,y,z], ctx)
            check_xyz_light = all([is_light(dot, ctx) for dot in [x,y,z]])
            check_xyz_alone = all([not all_close([x,y,z,dot], ctx) for dot in idxs if dot not in [x,y,z]])
            if (
                check_xyz_triangle
                and check_xyz_light
                and check_xyz_alone
            ):
                dots = frozenset([x,y,z])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.
Convert the above into JSON.
{{
    "speaker": "Them",
    "text": "Got a triangle of 3 light grey dots by itself.",
    "type": "New question.",
    "configs": "getsets(idxs, 3)",
    "configdots": "x,y,z",
    "newconfigs": "[0]",
    "newdots": "_",
    "constraints": [
        {{"name": "check_xyz_triangle", "code": "is_triangle([x,y,z], ctx)"}},
        {{"name": "check_xyz_light", "code": "all([is_light(dot, ctx) for dot in [x,y,z]])"}},
        {{"name": "check_xyz_alone", "code": "all([not all_close([x,y,z,dot], ctx) for dot in idxs if dot not in [x,y,z]])"}}
    ],
    "dots": "x,y,z",
    "select": "False"
}}

# You: Is the bottom one largest?
# def turn(state):
    # Follow up question.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c in permutations(config):
            check_a_bottom = a == get_bottom([a,b,c], ctx)
            check_a_largest = a == largest([a,b,c], ctx)
            if (
                check_a_bottom
                and check_a_largest
            ):
                dots = frozenset([a,b,c])
                if dots not in results:
                    results.add(dots)
                    orderedresults.append(dots)
                    parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.
Convert the above into JSON.
{{
    "speaker": "You",
    "text": "Is the bottom one largest?",
    "type": "Follow up question.",
    "configs": "state",
    "configdots": "a,b,c",
    "newconfigs": "[0]",
    "newdots": "_",
    "constraints": [
        {{"name": "check_a_bottom", "code": "a == get_bottom([a,b,c], ctx)"}},
        {{"name": "check_a_largest", "code": "a == largest([a,b,c], ctx)"}}
    ],
    "dots": "a,b,c",
    "select": "False"
}}
 
# Them: Is there a light dot to the right of them?
# def turn(state):
    # Follow up question.
    results = set()
    orderedresults = []
    parents = []
    for config in state:
        for a,b,c in permutations(config):
            for x, in get1idxs(idxs, exclude=[a,b,c]):
                check_x_right = all(is_right(x, y, ctx) for y in [a,b,c])
                check_x_light = is_light(x, ctx)
                if (
                    check_x_right
                    and check_x_light
                ):
                    dots = frozenset([a,b,c])
                    if dots not in results:
                        results.add(dots)
                        orderedresults.append(dots)
                        parents.append(config)
    return sort_state(orderedresults, parents, ctx, select=False)
state = turn(state)
# End.
Convert the above into JSON.
{{
    "speaker": "You",
    "text": "Is there a light dot to the right of them?",
    "type": "Follow up question, new dot.",
    "configs": "state",
    "configdots": "a,b,c",
    "newconfigs": "get1idxs(idxs, exclude=[a,b,c])",
    "newdots": "x,",
    "constraints": [
        {{"name": "check_x_right", "code": "all(is_right(x, y, ctx) for y in [a,b,c])"}},
        {{"name": "check_x_light", "code": "is_light(x, ctx)"}}
    ],
    "dots": "a,b,c,x",
    "select": "False"
}}

# You: Okay. <selection>.
def noop(state):
    # No op.
    return state
state = noop(state)
# End.
Convert the above into JSON.
{{
    "speaker": "You",
    "text": "Okay. <selection>.",
    "type": "No op.",
    "select": "False"
}}

{code}
Convert the above into JSON."""

blocks = ["""{
    "speaker": "Them",
    "text": "Got a triangle of 3 light grey dots by itself.",
    "type": "New question.",
    "configs": "getsets(idxs, 3)",
    "configdots": "x,y,z",
    "newconfigs": "[0]",
    "newdots": "_",
    "constraints": [
        {"name": "check_xyz_triangle", "code": "is_triangle([x,y,z], ctx)"},
        {"name": "check_xyz_light", "code": "all([is_light(dot, ctx) for dot in [x,y,z]])"},
        {"name": "check_xyz_alone", "code": "all([not all_close([x,y,z,dot], ctx) for dot in idxs if dot not in [x,y,z]])"}
    ],
    "dots": "x,y,z",
    "select": "False"
}"""]
for code in codes:
    prompt = construct_prompt(code)
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature = 0.2,
    )
    out = response["choices"][0]["message"]["content"]
    blocks.append(out)

outfile = Path("oc/promptdata/blocks-temp.txt")
with outfile.open("w") as f:
    f.write(f"[{','.join(blocks)}]")

blocks_json = json.loads(f"[{','.join(blocks)}]")
outfile = Path("oc/promptdata/blocks-temp.json")
with outfile.open("w") as f:
    json.dump(blocks_json, f)
