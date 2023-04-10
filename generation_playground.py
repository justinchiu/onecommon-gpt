import numpy as np
from pathlib import Path
import sys
import argparse
import minichain
import torch

from nltk import word_tokenize

from ocdata import get_data
from ocagent import Agent
from features import size_map3, color_map3, size_map5, color_map5
from features import size_color_descriptions, process_ctx, render

from eval import Recall

# fried arguments
oc_dir = Path("../onecommon/aaai2020/experiments")
#model_file = oc_dir / "expts/rel3_tsel_ref_dial_model_separate/jc-baseline/baseline/1/1_best.th"
model_file = oc_dir / "expts/rel3_tsel_ref_dial_model_separate/nov-15/plain-hierarchical-structured-recurrence/1/1_best.th"
detector_file = oc_dir / "serialized_models/markable_detector_with_dict_1.th"

# load scripts and initialize...better to make everything into libraries
sys.path.append(str(oc_dir.resolve()))
from engines.beliefs import BlankBeliefConstructor
from agent import RnnAgent
import utils
from cog_belief import CostBelief

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

    belief = CostBelief(                           
        7, view,                         
        absolute = True,
        num_size_buckets = num_buckets,
        num_color_buckets = num_buckets,
        use_diameter = False, 
        use_contiguity = False,
    )
    prior = belief.prior
    EdHs = belief.compute_EdHs(prior)
    planbool = belief.configs[EdHs.argmax()].astype(bool)
    plan = [{"target": planbool}]

    past = []

    print("plan")
    print([id for present, id in zip(planbool, dot_ids) if present])

    size_color = process_ctx(view, num_size_buckets=num_buckets, num_color_buckets=num_buckets)
    dots = size_color[planbool]
    descs = size_color_descriptions(dots, size_map=size_map3, color_map=color_map3)
    xy = view[planbool,:2]

    descstring = []
    for (size, color), (x,y) in zip(descs, xy):
        descstring.append(f"* A {size} and {color} dot (x={x:.2f},y={y:.2f})")

    with minichain.start_chain("tmp.txt") as backend:
        agent = Agent(backend, "codegen", "templateonly", "gpt-3.5-turbo")
        #agent = Agent(backend, "codegen", "templateonly", "gpt-4")
        out = agent.generate_text(plan, past, view)
        utt = out[0]
        words = word_tokenize(utt.lower().strip()) + ['<eos>']
        partner.read(
            words,
            detect_markables=True,
            dots_mentioned_num_markables = torch.tensor([0], device=device, dtype=torch.long),
        )

        print(planbool)
        print(partner.partner_ref_preds)

        fried_pred = partner.partner_ref_preds[-1][:,0]
        fried_rt_success = (planbool == fried_pred.any(0).cpu().numpy()).all()

        preds, past, extra = agent.resolve_reference(utt, past, view)
        gpt_rt_success = (planbool == preds).all(1).any()

        plans.append(plan)
        fried_preds.append(fried_pred)
        gpt_preds.append(preds)

        fried_successes += fried_rt_success
        gpt_successes += gpt_rt_success

metric = Recall("multilabel")

labels = [[x[0]["target"]] for x in plans]
fried_results = metric.compute(references=labels, predictions=[x.any(0)[None] for x in fried_preds])
gpt_results = metric.compute(references=labels, predictions=[x if len(x) > 0 else np.zeros((1,7)) for x in gpt_preds])

print("Fried successes")
print(fried_successes)
print(fried_results)

print("gpt successes")
print(gpt_successes)
print(gpt_results)
import pdb; pdb.set_trace()

