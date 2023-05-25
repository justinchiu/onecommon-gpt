import numpy as np
import itertools

from oc.gen.features import size_map3, color_map3, size_map5, color_map5
from oc.gen.features import size_color_descriptions, process_ctx, render
from oc.gen.features import new_vs_old_desc

from oc.prompt import Generate
from oc.prompt import GenerateScxy, GenerateTemplate

from oc.agent2.utils import StartPlan, FollowupPlan, SelectPlan, Speaker, State 
from oc.belief.belief_utils import get_config_idx

class WriterMixin:
    def __init__(self, backend, refres, gen, model):
        self.gen = gen
        if gen == "templateonly":
            pass
        else:
            raise ValueError

        super(WriterMixin, self).__init__()

    def write(self, force_action=None):
        state = self.states[-1]

        plan = self.plan(force_action=force_action)
        text, _, write_extra = self.generate_text(plan, state.past, self.ctx)

        self.states.append(State(
            belief_dist = state.belief_dist,
            plan = plan,
            past = state.past,
            speaker = Speaker.YOU,
            write_extra = write_extra,
            turn = state.turn+1,
            text = text,
        ))

        return text.split() + ["<eos>"]

    # GENERATION
    def generate_text(self, plan, past, view, info=None):
        if  self.gen == "templateonly":
            if isinstance(plan, SelectPlan):
                return self.generate_select(plan, past, view, info)
            elif isinstance(plan, StartPlan):
                return self.generate_new_config(plan, past, view, info)
            elif isinstance(plan, FollowupPlan):
                return self.generate_followup(plan, past, view, info)
            else:
                raise ValueError("Invalid plan")
        else:
            raise ValueError

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
        elif plan.confirmation == True:
            desc = f"Yes. {desc}"
        
        return desc, None, None

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
        elif plan.confirmation == False:
            out = f"No. {out}"

        return out, None, {"desc": descs, "position_desc": position_desc}


    def generate_select(self, plan, past, view, info=None):
        #newdot = plan.newdots.nonzero()[0].item()
        #olddots = plan.olddots.nonzero()[0].tolist()
        #descs, position_desc, olddescs = new_vs_old_desc(newdot, olddots, self.ctx, self.num_buckets)

        size_color = self.belief.size_color
        dot = size_color[plan.dots.nonzero()[0]]
        descs = size_color_descriptions(dot, size_map=size_map3, color_map=color_map3)
        #selectutt = f"Let's select the {descs[0][0]} size and {descs[0][1]} color one {position_desc} the {olddescs[0][0]} {olddescs[0][1]} one. <selection>"
        selectutt = f"Let's select the {descs[0][0]} size and {descs[0][1]} color one. <selection>"
        if plan.confirmation == True:
            selectutt = f"Yes. {selectutt}"
        elif plan.confirmation == False:
            selectutt = f"No. {selectutt}"
        return selectutt, None, None
