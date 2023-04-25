from dataclasses import dataclass
import numpy as np

@dataclass
class PlannerOutput:
    dots: np.ndarray
    newdot: int
    olddot: set[int]

class PlannerMixin:

    def plan(self, past, view, info=None):
        # was the previous plan confirmed or denied?
        previous_plan_confirmed = True
        return (
            self.plan_followup(past, view, info)
            if previous_plan_confirmed
            else self.plan_start(past, view, info)
        )

    def plan_start(self, past, view, info=None):
        pass

    def plan_followup(self, past, view, info=None):
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

