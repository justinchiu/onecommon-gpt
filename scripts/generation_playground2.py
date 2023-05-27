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
from oc.agent.utils import Action

from oc.belief.belief import CostBelief
from oc.gen.features import size_map3, color_map3, size_map5, color_map5
from oc.gen.features import size_color_descriptions, process_ctx, render

from functools import partial
from itertools import permutations                        

from oc.eval.eval import Recall

# fried arguments
oc_dir = Path("/home/justinchiu/research/onecommon/aaai2020/experiments")
oc_dir = Path("/Users/justinchiu/research/onecommon/aaai2020/experiments")
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

    belief_constructor = BlankBeliefConstructor()
    partner.feed_context(view.flatten().tolist(), belief_constructor)
    past = []

    with minichain.start_chain("tmp.txt") as backend:
        #agent = Agent(backend, "codegen", "templateonly", "gpt-3.5-turbo")
        
        #agent = Agent(backend, "codegen", "templateonly", "gpt-4")
        #agent = Agent(backend, "shortcodegen", "templateonly", "gpt-4")
        agent = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")
        reader = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")

        agent.feed_context(view.flatten().tolist(), belief_constructor)
        reader.feed_context(view.flatten().tolist(), belief_constructor)

        start_time = time.perf_counter()
        utt = agent.write()
        end_time = time.perf_counter()
        print(f"WRITE TIME: {end_time-start_time:0.4f} seconds for write")
        plan1 = agent.states[-1].plan

        start_time = time.perf_counter()
        reader.read(utt)
        end_time = time.perf_counter()
        print(f"READ TIME: {end_time-start_time:0.4f} seconds for read")

        words = word_tokenize(" ".join(utt).lower().strip()) + ['<eos>']
        partner.read(
            words,
            detect_markables=True,
            dots_mentioned_num_markables = torch.tensor([0], device=device, dtype=torch.long),
        )

        print(plan1)
        print(partner.partner_ref_preds)

        fried_pred = partner.partner_ref_preds[-1][:,0]
        fried_rt_success = (plan1.dots == fried_pred.any(0).cpu().numpy()).all()

        preds = reader.states[-1].plan.dots
        gpt_rt_success = (plan1.dots == preds).all()

        plans.append([plan1.dots])
        fried_preds.append(fried_pred)
        gpt_preds.append(preds)

        fried_successes += fried_rt_success
        gpt_successes += gpt_rt_success

        agent.read(["Them:", "Yes"])

        utt2 = agent.write(force_action=Action.FOLLOWUP)
        reader.read(utt2)
        plan2 = agent.states[-1].plan

        words = word_tokenize(" ".join(utt2).lower().strip()) + ['<eos>']
        partner.read(
            words,
            detect_markables=True,
            dots_mentioned_num_markables = torch.tensor([0], device=device, dtype=torch.long),
        )

        print(plan2.dots)
        print(partner.partner_ref_preds)

        fried_pred = partner.partner_ref_preds[-1][:,0]
        fried_rt_success = (plan2.dots == fried_pred.any(0).cpu().numpy()).all()

        preds = reader.states[-1].plan.dots
        gpt_rt_success = (plan2.dots == preds).all()

        plans2.append([plan2.dots])
        fried_preds2.append(fried_pred)
        gpt_preds2.append(preds)

        fried_successes2 += fried_rt_success
        gpt_successes2 += gpt_rt_success
        if len(preds) > 0 and not gpt_rt_success:
            metric = Recall("multilabel")
            print(plan2.dots)
            print(preds)
            #print(metric.compute(references=[[planbool2]], predictions=preds))
            import pdb; pdb.set_trace()

        agent.read(["Them:", "Yes"])

        # selection prompt
        # just select the new last one mentioned
        select_utt = agent.write(force_action=Action.SELECT)
        reader.read(select_utt)
        sel_plan = agent.states[-1].plan.dots
        sel_preds = reader.states[-1].plan.dots
        if sel_preds is not None and len(sel_preds) > 0:
            gpt_sel_rt_success = (sel_preds == sel_plan).all()
            gpt_successes3 += gpt_sel_rt_success
            if gpt_rt_success and not gpt_sel_rt_success:
                import pdb; pdb.set_trace()
        print(f"1 succeses {gpt_successes} / {example_idx+1}")
        print(f"2 succeses {gpt_successes2} / {example_idx+1}")
        print(f"3 succeses {gpt_successes3} / {example_idx+1}")


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
