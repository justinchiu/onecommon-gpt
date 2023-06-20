import time
import numpy as np
from pathlib import Path
import sys
import argparse
import minichain
import torch

from nltk import word_tokenize

from oc.ocdata import get_data
from oc.agent.agent import Agent
from oc.agent2.agent import Agent as Agent2
from oc.agent2.utils import Action
from oc.agent2.planner import StartPlan, FollowupPlan, SelectPlan

from oc.blip_agent.agent import BlipAgent

from oc.belief.belief import CostBelief
from oc.gen.features import size_map3, color_map3, size_map5, color_map5
from oc.gen.features import size_color_descriptions, process_ctx, render

from functools import partial
from itertools import permutations                        

from oc.eval.eval import Recall

# blip arguments
num_buckets = 3
#num_buckets = 5

num_examples = 25
#split = "train"
split = "valid"

traindata, validdata = get_data()
data = validdata if split == "valid" else traindata
data = data[:num_examples]

plans = []
blip_preds = []
gpt_preds = []
blip_successes = 0
gpt_successes = 0

plans2 = []
blip_preds2 = []
gpt_preds2 = []
blip_successes2 = 0
gpt_successes2 = 0

gpt_successes3 = 0

for example_idx, example in enumerate(data):
    # debug
    #if example_idx != 1: continue
    # /debug

    chatid = example["chat_id"]
    scenarioid = example["scenario_id"]
    print(scenarioid)
    print(chatid)

    view = example["context"]
    turns = example["dialogue"]
    referents = example["all_referents"]
    dot_ids = example["real_ids"]

    belief_constructor = None
    past = []

    with minichain.start_chain("tmp.txt") as backend:
        #agent = Agent(backend, "codegen", "templateonly", "gpt-3.5-turbo")
        
        #agent = Agent(backend, "codegen", "templateonly", "gpt-4")
        #agent = Agent(backend, "shortcodegen", "templateonly", "gpt-4")
        
        #agent = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")
        #reader = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")

        agent = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4-0613")
        reader = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4-0613")
        blip_reader = BlipAgent(backend)

        agent.feed_context(view.flatten().tolist(), belief_constructor)
        reader.feed_context(view.flatten().tolist(), belief_constructor)
        blip_reader.feed_context(view.flatten().tolist(), belief_constructor, scenario_id=scenarioid)

        start_time = time.perf_counter()
        utt = agent.write()
        end_time = time.perf_counter()
        print(f"WRITE TIME: {end_time-start_time:0.4f} seconds for write")
        plan1 = agent.states[-1].plan

        start_time = time.perf_counter()
        reader.read(utt)
        blip_reader.read(utt)
        end_time = time.perf_counter()
        print(f"READ TIME: {end_time-start_time:0.4f} seconds for read")

        words = word_tokenize(" ".join(utt).lower().strip()) + ['<eos>']

        print(plan1)

        blip_pred = partner.partner_ref_preds[-1][:,0]
        blip_rt_success = (plan1.dots == blip_pred.any(0).cpu().numpy()).all()

        preds = reader.states[-1].plan.dots
        gpt_rt_success = (preds.sum() == plan1.dots.sum()) and (plan1.dots == preds).all()

        plans.append([plan1.dots])
        blip_preds.append(blip_pred)
        gpt_preds.append(preds)

        blip_successes += blip_rt_success
        gpt_successes += gpt_rt_success

        if not gpt_rt_success:
            continue

        agent.read(["Them:", "Yes"])

        utt2 = agent.write(force_action=Action.FOLLOWUP)
        reader.read(utt2)
        plan2 = agent.states[-1].plan
        if not isinstance(plan2, FollowupPlan):
            import pdb; pdb.set_trace()

        words = word_tokenize(" ".join(utt2).lower().strip()) + ['<eos>']
        partner.read(
            words,
            detect_markables=True,
            dots_mentioned_num_markables = torch.tensor([0], device=device, dtype=torch.long),
        )

        print(plan2.dots)
        print(partner.partner_ref_preds)

        blip_pred = partner.partner_ref_preds[-1][:,0]
        blip_rt_success = (plan2.dots == blip_pred.any(0).cpu().numpy()).all()

        preds = reader.states[-1].plan.dots
        gpt_rt_success = (plan2.dots.sum() == preds.sum()) and (plan2.dots == preds).all()

        plans2.append([plan2.dots])
        blip_preds2.append(blip_pred)
        gpt_preds2.append(preds)

        blip_successes2 += blip_rt_success
        gpt_successes2 += gpt_rt_success
        if len(preds) > 0 and not gpt_rt_success:
            metric = Recall("multilabel")
            print(plan2.dots)
            print(preds)
            #print(metric.compute(references=[[planbool2]], predictions=preds))
            #import pdb; pdb.set_trace()

        agent.read(["Them:", "Yes"])

        # selection prompt
        # just select the new last one mentioned
        select_utt = agent.write(force_action=Action.SELECT)
        reader.read(select_utt)
        sel_plan = agent.states[-1].plan.dots
        sel_preds = reader.states[-1].plan.dots if reader.states[-1].plan is not None else None
        if sel_preds is not None and len(sel_preds) > 0:
            gpt_sel_rt_success = (sel_preds.sum() == sel_plan.sum()) and (sel_preds == sel_plan).all()
            gpt_successes3 += gpt_sel_rt_success
            if gpt_rt_success and not gpt_sel_rt_success:
                #import pdb; pdb.set_trace()
                pass
        print(f"1 succeses {gpt_successes} / {example_idx+1}")
        print(f"2 succeses {gpt_successes2} / {example_idx+1}")
        print(f"3 succeses {gpt_successes3} / {example_idx+1}")


metric = Recall("multilabel")

labels = plans
blip_results = metric.compute(references=labels, predictions=[x.any(0)[None] for x in blip_preds])
gpt_results = metric.compute(references=labels, predictions=[x if len(x) > 0 else np.zeros((1,7)) for x in gpt_preds])

print("Blip successes")
print(blip_successes)
print(blip_results)

print("gpt successes")
print(gpt_successes)
print(gpt_results)

labels = plans2
blip_results = metric.compute(references=labels, predictions=[x.any(0)[None] for x in blip_preds2])
gpt_results = metric.compute(references=labels, predictions=[x if len(x) > 0 else np.zeros((1,7)) for x in gpt_preds2])

print("Blip successes 2")
print(blip_successes2)
print(blip_results)

print("gpt successes 2")
print(gpt_successes2)
print(gpt_results)

print("gpt_successes 3")
print(gpt_successes3)
