import numpy as np

from oc.agent.utils import Plan

def new_and_old_dots(plan, history):
    if len(history) > 0:
        import pdb; pdb.set_trace()
        plans = [x.plan for x in history]
    else:
        return plan, np.zeros(7, dtype=bool)

class PlannerMixin:

    def update_belief(self, response):
        import pdb; pdb.set_trace()
        plan = self.plans[-1]
        self.belief_dist = belief.posterior(
            self.belief_dist,
            plan.astype(int),
            response,
        )

    def plan(self):
        # was the previous plan confirmed or denied?
        previous_plan_confirmed = True
        plan = (
            self.plan_followup(past, view, info)
            if previous_plan_confirmed
            else self.plan_start(past, view, info)
        )
        self.plans.append(plan)
        return plan

    def plan_start(self):
        EdHs = self.belief.compute_EdHs(self.belief_dist)
        planbool = self.belief.configs[EdHs.argmax()].astype(bool)
        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)
        new, old = new_and_old_dots(planbool, self.plans)
        plan = Plan(
            dots = planbool,
            newdots = new,
            olddots = old,
        )
        return plan

    def plan_followup(self):
        EdHs = self.belief.compute_EdHs(self.belief_dist)
        # mask out plans that don't have the desired configs
        EdHs_mask = [
            any(
                set(idxs).issubset(set(config.nonzero()[0]))
                for idxs in plan_idxs
            )
            for config in belief.configs
        ]
        EdHs *= EdHs_mask
        planbool2 = belief.configs[EdHs.argmax()].astype(bool)

        feats2 = belief.get_feats(planbool2)
        plan_idxs2 = belief.resolve_utt(*feats2)

