import numpy as np
from pathlib import Path
import sys
import argparse
import minichain
import torch

from nltk import word_tokenize

from oc.ocdata import get_data
from oc.agent.agent import Agent
from oc.belief.belief import CostBelief
from oc.gen.features import size_map3, color_map3, size_map5, color_map5
from oc.gen.features import size_color_descriptions, process_ctx, render

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

from functools import partial
from itertools import permutations                        

from oc.eval.eval import Recall

# fried arguments
oc_dir = Path("/home/justinchiu/research/onecommon/aaai2020/experiments")
#oc_dir = Path("/Users/justinchiu/research/onecommon/aaai2020/experiments")
#model_file = oc_dir / "expts/rel3_tsel_ref_dial_model_separate/jc-baseline/baseline/1/1_best.th"
model_file = oc_dir / "expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1/1_best.th"
detector_file = oc_dir / "serialized_models/markable_detector_with_dict_1.th"

# load scripts and initialize...better to make everything into libraries
sys.path.append(str(oc_dir.resolve()))
print(oc_dir.resolve())
from engines.beliefs import BlankBeliefConstructor
from agent import RnnAgent
import utils

markable_detector = utils.load_model(detector_file, prefix_dir=None, map_location="cpu")
markable_detector.eval()

model = utils.load_model(model_file, prefix_dir=None, map_location="cpu")
model_args = model.args
model.eval()

if torch.cuda.is_available():
    markable_detector.cuda()
    model.cuda()
    model.device = "cuda"
    markable_detector.device = "cuda"
    device = "cuda"
else:
    model.device = "cpu"
    markable_detector.device = "cpu"
    device = "cpu"

# dummy args
parser = argparse.ArgumentParser()
RnnAgent.add_args(parser)
agent_args = parser.parse_args()

# set args
agent_args.language_rerank = True
agent_args.next_mention_reranking = True
agent_args.language_beam_keep_all_finished = True
agent_args.reranking_confidence = True
agent_args.language_beam_size = 16
agent_args.next_mention_reranking_k = 4
agent_args.next_mention_reranking_max_mentions = 4

merged_args = argparse.Namespace(**utils.merge_dicts(vars(agent_args), vars(model.args)))

merged_args.cuda = torch.cuda.is_available()

# make agent
partner = RnnAgent(
    model,
    merged_args,
    name="Alice",
    train=False,
    markable_detector=markable_detector,
)
# on to generation

num_buckets = 3
#num_buckets = 5

chat_id_list = [
    "C_0dd19b44543141beb1737f391f2a1899",
]

use_chat_id_list = False
num_examples = 25
#split = "train"
split = "valid"

traindata, validdata = get_data()
data = validdata if split == "valid" else traindata
if use_chat_id_list:
    data = [
        ex for ex in data
        if ex["chat_id"] in chat_id_list
    ]
data = data[:num_examples]

plans = []
fried_preds = []
gpt_preds = []
fried_successes = 0
gpt_successes = 0

plans2 = []
fried_preds2 = []
gpt_preds2 = []
fried_successes2 = 0
gpt_successes2 = 0

gpt_successes3 = 0

for example in data:
    chatid = example["chat_id"]
    scenarioid = example["scenario_id"]
    print(scenarioid)
    print(chatid)

    view = example["context"]
    turns = example["dialogue"]
    referents = example["all_referents"]
    dot_ids = example["real_ids"]

    belief_constructor = BlankBeliefConstructor()
    partner.feed_context(view.flatten().tolist(), belief_constructor)
    past = []

    with minichain.start_chain("tmp.txt") as backend:
        #agent = Agent(backend, "codegen", "templateonly", "gpt-3.5-turbo")
        agent = Agent(backend, "codegen", "templateonly", "gpt-4")

        agent.feed_context(view.flatten().tolist(), belief_constructor)

        #plan1 = agent.plan_start(agent.plans, view)
        #out = agent.generate_text(plan1, past, view)
        #utt = out[0]

        utt = agent.write()
        import pdb; pdb.set_trace()

        words = word_tokenize(utt.lower().strip()) + ['<eos>']
        partner.read(
            words,
            detect_markables=True,
            dots_mentioned_num_markables = torch.tensor([0], device=device, dtype=torch.long),
        )

        print(plan1)
        print(partner.partner_ref_preds)

        fried_pred = partner.partner_ref_preds[-1][:,0]
        fried_rt_success = (plan1.dots == fried_pred.any(0).cpu().numpy()).all()

        preds, past, extra = agent.resolve_reference(utt, past, view)
        radii = [get_minimum_radius(pred.nonzero()[0], view) for pred in preds]
        # sort by radii
        sorted_preds = preds[np.argsort(radii)]
        gpt_rt_success = (plan1.dots == preds).all(1).any()

        plans.append([plan1.dots])
        fried_preds.append(fried_pred)
        gpt_preds.append(sorted_preds)

        assert (preds == sorted_preds).all()

        fried_successes += fried_rt_success
        gpt_successes += gpt_rt_success

        agent.read(["Them:", "Yes"])
        import pdb; pdb.set_trace()
        plan2 = agent.plan_followup(agent.plans, view)

        posterior = belief.posterior(prior, planbool.astype(int), 1)
        EdHs = belief.compute_EdHs(posterior)
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

        print(plan_idxs)
        print(plan_idxs2)
        dotsets = [set(x) for x in plan_idxs]
        dotsets2 = [set(x) for x in plan_idxs2]
        import itertools
        setpairs = list(itertools.product(dotsets, dotsets2))
        smalldiffs = [(x,y) for x,y in setpairs if len(y.difference(x)) == 1]
        radii = [get_minimum_radius(list(y), view) for x,y in smalldiffs]
        smallest_idx = np.argmin(radii)
        olddotset = smalldiffs[smallest_idx][0]
        newdotset = smalldiffs[smallest_idx][1]
        newdot = list(newdotset.difference(olddotset))
        olddots = list(olddotset)
        newdots = list(newdotset)

        import pdb; pdb.set_trace()

        planbool2 = np.zeros(7, dtype=bool)
        planbool2[newdots] = 1

        ctx = view
        plan2 = [{"target": planbool2}]

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

        dots2 = size_color[newdot]
        descs = size_color_descriptions(dots2, size_map=size_map3, color_map=color_map3)

        #newutt = f"Is there a {descs[0][0]} size and {descs[0][1]} color dot {position_desc} that?"
        newutt = f"Is there a {descs[0][0]} size and {descs[0][1]} color dot {position_desc} those?"

        words = word_tokenize(newutt.lower().strip()) + ['<eos>']
        partner.read(
            words,
            detect_markables=True,
            dots_mentioned_num_markables = torch.tensor([0], device=device, dtype=torch.long),
        )

        print(planbool2)
        print(partner.partner_ref_preds)

        fried_pred = partner.partner_ref_preds[-1][:,0]
        fried_rt_success = (planbool == fried_pred.any(0).cpu().numpy()).all()

        preds, past, extra = agent.resolve_reference(newutt, past, view)
        radii = [get_minimum_radius(pred.nonzero()[0], ctx) for pred in preds]
        # sort by radii
        sorted_preds = preds[np.argsort(radii)]
        gpt_rt_success = (planbool2 == preds).all(1).any()

        plans2.append([planbool2])
        fried_preds2.append(fried_pred)
        gpt_preds2.append(sorted_preds)

        fried_successes2 += fried_rt_success
        gpt_successes2 += gpt_rt_success
        if len(preds)> 0 and not gpt_rt_success:
            metric = Recall("multilabel")
            print(planbool2)
            print(preds)
            #print(metric.compute(references=[[planbool2]], predictions=preds))
            import pdb; pdb.set_trace()

        # selection prompt
        posterior2 = belief.posterior(posterior, planbool2.astype(int), 1)

        # just select the new last one mentioned

        selectutt = f"Let's select the {descs[0][0]} size and {descs[0][1]} color one on the {position_desc}."
        sel_preds, past, extra = agent.resolve_reference(selectutt, past, view)
        # sort sel_preds by minimum radius of PREVIOUS plans
        if len(sel_preds) > 0:
            gpt_sel_rt_success = sel_preds[np.argmin(radii)].nonzero()[0].item() == newdot[0]
            gpt_successes3 += gpt_sel_rt_success
            if gpt_rt_success and not gpt_sel_rt_success:
                import pdb; pdb.set_trace()


metric = Recall("multilabel")

labels = plans
fried_results = metric.compute(references=labels, predictions=[x.any(0)[None] for x in fried_preds])
gpt_results = metric.compute(references=labels, predictions=[x if len(x) > 0 else np.zeros((1,7)) for x in gpt_preds])

print("Fried successes")
print(fried_successes)
print(fried_results)

print("gpt successes")
print(gpt_successes)
print(gpt_results)

labels = plans2
fried_results = metric.compute(references=labels, predictions=[x.any(0)[None] for x in fried_preds2])
gpt_results = metric.compute(references=labels, predictions=[x if len(x) > 0 else np.zeros((1,7)) for x in gpt_preds2])

print("Fried successes 2")
print(fried_successes2)
print(fried_results)

print("gpt successes 2")
print(gpt_successes2)
print(gpt_results)

print("gpt_successes 3")
print(gpt_successes3)
