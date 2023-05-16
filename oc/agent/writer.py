import numpy as np
import itertools

from oc.gen.features import size_map3, color_map3, size_map5, color_map5
from oc.gen.features import size_color_descriptions, process_ctx, render
from oc.gen.features import new_vs_old_desc

from oc.prompt import Generate
from oc.prompt import GenerateScxy, GenerateTemplate


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

    def write(self, force_action=None):
        plan = self.plan(force_action=force_action)
        text, _, write_extra = self.generate_text(plan, self.past, self.ctx)

        youtext = f"You: {text}"
        preds, past, read_extra = self.resolve_reference(youtext, self.past, self.ctx)

        # TODO: wrap state update in function
        self.past = past
        self.preds.append(preds)
        self.confirmations.append(None)
        self.write_extras.append(write_extra)
        self.read_extras.append(read_extra)

        return text.split() + ["<eos>"]

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
        if plan.confirmation == False:
            desc = f"No. {desc}"
        
        return desc, past + [desc], None

    # for rule-based generation with simple coref
    def generate_new_config(self, plan, past, view, info=None):
        return self.generate_text_template_only(plan, past, view, info)

    def generate_followup(self, plan, past, view, info=None):
        newdot = plan.newdots.nonzero()[0].item()
        olddots = list(plan.olddots.nonzero()[0])

        descs, position_desc, olddescs = new_vs_old_desc(newdot, olddots, self.ctx, self.num_buckets)

        out = f"Is there a {descs[0][0]} size and {descs[0][1]} color dot {position_desc} those?"
        if plan.confirmation == True:
            out = f"Yes. {out}"

        return out, past + [out], {"desc": descs, "position_desc": position_desc}


    def generate_select(self, plan, past, view, info=None):
        newdot = plan.newdots.nonzero()[0].item()
        olddots = plan.olddots.nonzero()[0].tolist()
        descs, position_desc, olddescs = new_vs_old_desc(newdot, olddots, self.ctx, self.num_buckets)

        #selectutt = f"Let's select the {descs[0][0]} size and {descs[0][1]} color one {position_desc} the {olddescs[0][0]} {olddescs[0][1]} one. <selection>"
        selectutt = f"Let's select the {descs[0][0]} size and {descs[0][1]} color one. <selection>"
        return selectutt, past + [selectutt], None
