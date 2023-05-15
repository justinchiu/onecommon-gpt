
def plan_select_old(self, belief_dist, plans):
    """
    if True in pred_successes:
        # find the last partner plan we see
        ridx = list(reversed(pred_successes)).index(True)
        idx = len(self.confirmations) - ridx - 1
        import pdb; pdb.set_trace()
    else:
        # otherwise find the last confirmed plan
        ridx = list(reversed(self.confirmations)).index(True)
        idx = len(self.confirmations) - ridx - 1
        lastplan = self.plans[idx-1]
        dots = lastplan.dots
        olddots = lastplan.olddots
    """

    planbool, new, old, new_idxs = None, None, None, None
    if len(self.preds) < 2:
        marginals = self.belief.marginals(belief_dist)
        assert self.preds[0].sum() > 0
        selectdot = marginals.argmax()

        new = np.zeros(7, dtype=bool)
        new[selectdot] = True

        planbool = self.preds[0][0]
        old = planbool.copy()
        old[selectdot] = False

        feats = self.belief.get_feats(planbool)
        new_idxs = self.belief.resolve_utt(*feats)
    else:
        pred_successes = [x.sum() > 0 for x in self.preds]
        revsuc = list(reversed(pred_successes)) 
        ridx1 = revsuc.index(True)
        ridx2 = revsuc[ridx1+1:].index(True) + ridx1 + 1

        idx1 = len(self.preds) - ridx1 - 1
        idx2 = len(self.preds) - ridx2 - 1
        dots = self.preds[idx1][0]
        olddots = self.preds[idx2][0]

        feats = self.belief.get_feats(dots)
        plan_idxs = self.belief.resolve_utt(*feats)

        # choose the second last confirmed dot
        prev_feats = self.belief.get_feats(olddots)
        prev_plan_idxs = self.belief.resolve_utt(*prev_feats)

        new_idxs, old_idxs = idxs_to_dots(prev_plan_idxs, plan_idxs, self.ctx)

        # disambiguated
        planbool = np.zeros(7, dtype=bool)
        planbool[new_idxs] = 1

        old = np.zeros(7, dtype=bool)
        old[old_idxs] = 1

        new = planbool & ~old

    confirmation = self.should_confirm()

    plan = Plan(
        dots = planbool,
        newdots = new,
        olddots = old,
        plan_idxs = new_idxs,
        should_select = True,
        confirmation = confirmation,
        info_gain = None,
    )
    return plan

