import numpy as np
import itertools

from oc.agent2.utils import Plan, StartPlan, FollowupPlan, SelectPlan, Action, Qtypes
from oc.belief.belief_utils import get_config_idx
from oc.fns.spatial import get_minimum_radius

def new_and_old_dots(plan, history):
    if len(history) > 0:
        plans = [x.dots for x in history if x is not None]
        if len(plans) == 0:
            return plan, None
        plans = np.any(plans, axis=0)
        return plan & ~plans, plans
    else:
        return plan, None

def idxs_to_dots(plan_idxs, plan_idxs2, view):
    dotsets = [set(x) for x in plan_idxs]
    dotsets2 = [set(x) for x in plan_idxs2]
    setpairs = list(itertools.product(dotsets, dotsets2))
    smalldiffs = [(x,y) for x,y in setpairs if len(y.difference(x)) == 1]
    radii = [get_minimum_radius(list(y), view) for x,y in smalldiffs]
    smallest_idx = np.argmin(radii)
    olddotset = smalldiffs[smallest_idx][0]
    newdotset = smalldiffs[smallest_idx][1]
    newdot = list(newdotset.difference(olddotset))
    olddots = list(olddotset)
    newdots = list(newdotset)
    return newdots, olddots

class PlannerMixin:

    def update_belief(self, belief_dist, dots, response) -> None:
        return self.belief.posterior(
            belief_dist,
            dots.astype(int),
            response,
        )

    def plan(self, force_action=None):
        state = self.states[-1]

        select_plan = self.plan_select(self.states, force=force_action == Action.SELECT)
        followup_plan = self.plan_followup(self.states)
        start_plan = self.plan_start(self.states)

        plan = None
        if force_action == Action.SELECT:
            plan = select_plan
        elif force_action == Action.FOLLOWUP:
            plan = followup_plan
        elif force_action == Action.START:
            plan = start_plan
            # end forced 
        elif select_plan is not None:
            plan = select_plan
        elif followup_plan is not None and (
            # you confirm what they said last
            followup_plan.confirmation == True
            # or you follow up on your last plan after they only say "yes"
            or (
                len(self.states) > 2
                and self.states[-1].plan == None
                and self.states[-2].plan.confirmed
            )
        ):
            plan = followup_plan
        elif followup_plan is not None and followup_plan.info_gain > start_plan.info_gain:
            plan = followup_plan
        else:
            plan = start_plan
        # DBG
        #if followup_plan is not None:
        #    import pdb; pdb.set_trace()
        # /DBG
        #
        #
        return plan

    def plan_start(self, states):
        belief_dist = states[-1].belief_dist

        EdHs = self.belief.compute_EdHs(belief_dist)

        # TODO: maybe mask out all configs that arent size 2?
        EdHs_mask = self.belief.configs.sum(-1) == 2
        repeat_mask = self.get_repeat_mask(states)
        EdHs *= EdHs_mask * repeat_mask

        idx = EdHs.argmax()
        info_gain = EdHs[idx]
        planbool = self.belief.configs[idx].astype(bool)

        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)

        confirmation = self.should_confirm(states)

        plan = StartPlan(
            dots = planbool,
            config_idx = get_config_idx(planbool, self.belief.configs),
            feats = feats,
            plan_idxs = plan_idxs,
            all_dots = planbool[None], # TODO: not sure this works
            confirmation = confirmation,
            info_gain = info_gain,
            confirmed = None, # fill in future turn
            qtype = Qtypes.START,
            new_dots = planbool.sum().item(),
        )
        return plan

    def plan_followup(self, states):
        if len(states) <= 1:
            return None

        belief_dist = states[-1].belief_dist

        EdHs = self.belief.compute_EdHs(belief_dist)

        # choose plan to follow up on
        dots, reference_turn = self.get_last_confirmed_dots(states)

        if dots is None:
            return None

        config_mask = (self.belief.configs & dots).sum(-1) == dots.sum()
        inclusion_prob = (belief_dist * config_mask).sum()
        print(f"Followup config inclusion prob: {inclusion_prob}")

        # mask out plans that aren't dots + 1 new
        EdHs_mask = (self.belief.configs & dots).sum(-1) == dots.sum()
        repeat_mask = self.get_repeat_mask(states)
        newdot_mask = self.belief.configs.sum(-1) == dots.sum() + 1
        EdHs *= EdHs_mask * repeat_mask * newdot_mask
        idx = EdHs.argmax()
        info_gain = EdHs[idx]
        planbool = self.belief.configs[idx].astype(bool)

        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)

        prev_feats = self.belief.get_feats(dots)
        prev_plan_idxs = self.belief.resolve_utt(*prev_feats)

        new_idxs, old_idxs = idxs_to_dots(prev_plan_idxs, plan_idxs, self.ctx)

        # disambiguated
        planbool2 = np.zeros(7, dtype=bool)
        planbool2[new_idxs] = 1

        old = np.zeros(7, dtype=bool)
        old[old_idxs] = 1

        new = planbool2 & ~old

        confirmation = self.should_confirm(states)

        plan = FollowupPlan(
            dots = planbool2,
            config_idx = get_config_idx(planbool, self.belief.configs),
            feats = feats,
            plan_idxs = plan_idxs,
            all_dots = planbool2[None], # TODO: not sure this works
            newdots = new,
            olddots = old,
            info_gain = info_gain,
            confirmation = confirmation,
            confirmed = None,
            reference_turn = reference_turn,
            qtype = Qtypes.FNEW,
            new_dots = new.sum().item(),
        )
        return plan

    def plan_select(self, states, force=False):
        if len(states) <= 1: return None

        belief_dist = states[-1].belief_dist
        confirmation = self.should_confirm(states)

        marginals = self.belief.marginals(belief_dist)
        if (
            (marginals < self.belief_threshold).all()
            and len(states) > self.min_turns
        ) or len(states) >= self.max_turns:
            # give up and return most likely one
            planbool = np.zeros(7, dtype=bool)
            planbool[marginals.argmax()] = 1
            feats = self.belief.get_feats(planbool)
            plan_idxs = self.belief.resolve_utt(*feats)
            plan = SelectPlan(
                dots = planbool,
                feats = feats,
                plan_idxs = plan_idxs,
                all_dots = planbool[None], # TODO: not sure this works
                olddots = None,
                config_idx = get_config_idx(planbool, self.belief.configs),
                confirmation = confirmation,
                confirmed = None,
                reference_turn = None,
                info_gain = None,
                qtype = Qtypes.SELECT,
                new_dots = 0,
            )
            return plan


        dots, reference_turn = self.get_last_confirmed_dots(states)
        if dots is None: return None

        config_mask = (self.belief.configs & dots).sum(-1) == dots.sum()
        inclusion_prob = (belief_dist * config_mask).sum()

        marginals = self.belief.marginals(belief_dist) * dots

        threshold = self.belief_threshold
        mask = marginals > threshold

        num_sure_dots = mask.sum()
        if not force and num_sure_dots < 2:
            # Do not select if there are less than 2 we are sure about
            return None

        # maybe need saliency prior instead of highest prob?
        #import pdb; pdb.set_trace()

        dot_order = np.argsort(marginals)[::-1]
        anchor_dot = dot_order[0]

        """
        size_color = self.belief.size_color[dots]
        nomatch = ((size_color != size_color[:,None]).all(-1) + np.eye(dots.sum())).all(-1)
        nomatch_ordering = np.argsort(marginals[dots] * nomatch)[::-1]
        anchor_dot = dots.nonzero()[0][nomatch_ordering[0]]
        """

        # talk about at most 2 other dots
        #num_sure_dots = min(num_sure_dots, 2)
        #aux_dots = dot_order[1:num_sure_dots+1]

        # just one other dot
        #aux_dots = dot_order[1]

        newdots = np.zeros(7, dtype=bool)
        newdots[anchor_dot] = True
        olddots = np.zeros(7, dtype=bool)
        #olddots[aux_dots] = True

        planbool = newdots + olddots

        feats = self.belief.get_feats(planbool)
        plan_idxs = self.belief.resolve_utt(*feats)

        plan = SelectPlan(
            dots = planbool,
            feats = feats,
            plan_idxs = plan_idxs,
            all_dots = planbool[None], # TODO: not sure this works
            olddots = dots,
            config_idx = get_config_idx(planbool, self.belief.configs),
            confirmation = confirmation,
            confirmed = None,
            reference_turn = reference_turn,
            info_gain = None,
            qtype = Qtypes.SELECT,
            new_dots = 0,
        )
        return plan

    def choose(self):
        confirmed_or_select = [
            state.plan.dots for state in reversed(self.states)
            if state.plan is not None
            and (state.plan.confirmed or isinstance(state.plan, SelectPlan))
            and state.plan.dots.sum() == 1
        ]
        return (
            confirmed_or_select[0].nonzero()[0].item()
            if confirmed_or_select
            else self.belief.marginals(self.states[-1].belief_dist).argmax()
        )

    def should_select(self, states):
        belief_dist = states[-1].belief_dist
        max_belief = self.belief.marginals(belief_dist).max()
        select = max_belief > self.belief_threshold
        return select

    def should_confirm(self, states):
        if states[-1].plan is None:
            return None
        else:
            return states[-1].plan.confirmed

    def get_last_confirmed_dots(self, states):
        belief_dist = states[-1].belief_dist
        confirmed = [
            (state.plan.dots, state.turn) for state in reversed(states)
            if (
                state.plan is not None
                and state.plan.confirmed == True
                #and self.belief.p_response(belief_dist, state.plan.dots)[1] > 0.5
                #and self.belief.p_response(belief_dist, state.plan.dots)[1] > 0.6
            )
        ]
        return confirmed[0] if confirmed else (None, None)

    def get_last_confirmed_all_dots(self, states):
        confirmed = [
            (state.plan.all_dots, state.turn) for state in reversed(states)
            if state.plan is not None and state.plan.confirmed == True
        ]
        return confirmed[0] if confirmed else (None, None)

    def get_repeat_mask(self, states):
        # kill any configs that have already been asked
        mask = np.ones(self.belief.configs.shape[0], dtype=float)
        for state in states:
            if state.plan is not None and state.plan.config_idx is not None:
                mask[state.plan.config_idx] = 0
        return mask
