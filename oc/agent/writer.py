import numpy as np
import itertools

from oc.gen.features import size_map3, color_map3, size_map5, color_map5
from oc.gen.features import size_color_descriptions, process_ctx, render

from oc.prompt import Generate
from oc.prompt import GenerateScxy, GenerateTemplate

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


class WriterMixin:
    def __init__(self, backend, refres, gen, model):
        self.gen = gen
        if gen == "sc":
            self.generate = Generate(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=512,
            ))
        elif gen == "scxy":
            self.generate = GenerateScxy(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=512,
            ))
        elif gen == "template":
            self.generate = GenerateTemplate(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=1024,
            ))
        elif gen == "templateonly":
            pass
        else:
            raise ValueError

        super(WriterMixin, self).__init__()

    def write(self):
        plan = self.plan()
        text, _, write_extra = self.generate_text(plan, self.past, self.ctx)

        youtext = f"You: {text}"
        preds, past, read_extra = self.resolve_reference(youtext, self.past, self.ctx)

        # TODO: wrap state update in function
        self.past = past
        self.preds.append(preds)
        self.confirmations.append(None)
        self.write_extras.append(write_extra)
        self.read_extras.append(read_extra)

        return text

    # GENERATION
    def generate_text(self, plan, past, view, info=None):
        if self.gen == "sc":
            return self.generate_text_sc(plan, past, view, info)
        if self.gen == "scxy":
            return self.generate_text_scxy(plan, past, view, info)
        elif self.gen == "template":
            return self.generate_text_template(plan, past, view, info)
        elif self.gen == "templateonly":
            if plan.should_select:
                return self.generate_select(plan, past, view, info)
            if plan.olddots is None:
                return self.generate_new_config(plan, past, view, info)
            else:
                return self.generate_followup(plan, past, view, info)
            #return self.generate_text_template_only(plan, past, view, info)
        else:
            raise ValueError

    def generate_text_sc(self, plan, past, view, info=None):
        raise NotImplementedError
        # process plan
        refs = [r["target"] for r in plan]
        size_color = process_ctx(view)
        dots = size_color[np.array(refs).any(0)]
        descs = size_color_descriptions(dots)
        descstring = []
        for size, color in descs:
            descstring.append(f"* A {size} and {color} dot")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        #print("INPUT")
        #print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out], None

    def generate_text_scxy(self, plan, past, view, info=None):
        raise NotImplementedError
        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)

        size_color = process_ctx(view)
        dots = size_color[plan]
        descs = size_color_descriptions(dots)
        xy = view[plan,:2]

        descstring = []
        for (size, color), (x,y) in zip(descs, xy):
            descstring.append(f"* A {size} and {color} dot (x={x:.2f},y={y:.2f})")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        print("INPUT")
        print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out], None

    def generate_text_template(self, plan, past, view, info=None):
        raise NotImplementedError
        if len(plan) == 0:
            # no references...
            return "okay", past + ["okay"]

        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)
        desc = render(plan, view)

        kwargs = dict(plan=desc, past="\n".join(past))
        #print("INPUT")
        #print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out], None

    def generate_text_template_only(self, plan, past, view, info=None):
        if plan.dots.sum() == 0:
            # no references...
            return "okay", past + ["okay"]

        # process plan
        desc = render(plan.dots, view, num_buckets=3)
        print(desc)
        # not sure if those last two are needed
        # TODO: only return desc, since plan history is stored in agent.plan
        
        # insert confirmation
        if plan.confirmation is False:
            desc = f"No. {desc}"
        
        return desc, past + [desc], None

    # for rule-based generation with simple coref
    def generate_new_config(self, plan, past, view, info=None):
        return self.generate_text_template_only(plan, past, view, info)

    def generate_followup(self, plan, past, view, info=None):
        ctx = self.ctx

        newdot = plan.newdots.nonzero()[0].item()
        olddots = list(plan.olddots.nonzero()[0])

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
            import pdb; pdb.set_trace()
            raise ValueError

        size_color = process_ctx(
            self.ctx,
            num_size_buckets=self.num_buckets,
            num_color_buckets=self.num_buckets,
        )
        dots2 = size_color[[newdot]]
        descs = size_color_descriptions(dots2, size_map=size_map3, color_map=color_map3)

        out = f"Is there a {descs[0][0]} size and {descs[0][1]} color dot {position_desc} those?"
        if plan.confirmation is True:
            out = f"Yes. {out}"

        return out, past + [out], {"desc": descs, "position_desc": position_desc}


    def generate_select(self, plan, past, view, info=None):
        # TODO: this is probably going to fail. worry about it later
        descs = agent.write_extras[-2]["desc"]
        position_desc = agent.write_extras[-2]["position_desc"]
        selectutt = f"Let's select the {descs[0][0]} size and {descs[0][1]} color one {position_desc} those."
        return selectutt, past + [selectutt], None
