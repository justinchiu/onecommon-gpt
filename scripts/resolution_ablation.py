import time
import numpy as np
from pathlib import Path
import sys
import argparse
import minichain
import torch
import json

from nltk import word_tokenize

from oc.ocdata import get_data
from oc.agent.agent import Agent
from oc.agent2.agent import Agent as Agent2
from oc.agent2.utils import Action
from oc.agent2.planner import StartPlan, FollowupPlan, SelectPlan

from oc.belief.belief import CostBelief
from oc.gen.features import size_map3, color_map3, size_map5, color_map5
from oc.gen.features import size_color_descriptions, process_ctx, render

from functools import partial
from itertools import permutations                        

from oc.eval.eval import Recall


chats = Path("/home/justinchiu/research/onecommon/webapp/gpt_experiments_turk/blah.json")
#chats = Path("/Users/justinchiu/research/onecommon/webapp/gpt_experiments_turk/blah.json")
with chats.open("r") as f:
    chats = json.load(f)

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
from utils import ContextGenerator

#ctx_gen = ContextGenerator("/Users/justinchiu/research/onecommon/aaai2020/experiments/data/onecommon/shared_4.txt")
ctx_gen = ContextGenerator("/home/justinchiu/research/onecommon/aaai2020/experiments/data/onecommon/shared_4.txt")
boards = {
    x[0][0]: x
    for x in ctx_gen.ctxs
}

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


subsets = [
    #np.random.choice(15, size=10, replace=False) for _ in range(5)
    #np.random.choice(15, size=5, replace=False) for _ in range(5)
    np.random.choice(15, size=3, replace=False) for _ in range(5)
    #np.random.choice(15, size=1, replace=False) for _ in range(5)
]

accuracies = [[] for _ in range(5)]
counter = 0

for example_idx, example in enumerate(chats):
    # debug
    #if example_idx != 1: continue
    # /debug

    chatid = example["chat_id"]
    scenarioid = example["scenario_id"]
    print(scenarioid)
    print(chatid)

    board = boards[scenarioid]

    dialogue = example["dialogue"]
    first_agent = dialogue[0][0]
    if example["agent_types"][str(first_agent)] != "human":
        continue

    if len(dialogue) < 2:
        continue

    us_agent = dialogue[1][0]

    view = np.array(board[1 if us_agent == 0 else 2], dtype=np.float64)
    first_turn = dialogue[0][1]

    belief_constructor = BlankBeliefConstructor()
    partner.feed_context(view.flatten().tolist(), belief_constructor)
    past = []

    with minichain.start_chain("tmp.txt") as backend:
        #agent = Agent(backend, "codegen", "templateonly", "gpt-3.5-turbo")
        
        #agent = Agent(backend, "codegen", "templateonly", "gpt-4")
        #agent = Agent(backend, "shortcodegen", "templateonly", "gpt-4")
        
        #agent = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")
        #reader = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")

        agent_15 = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4-0613")
        agent_15.feed_context(view.flatten().tolist(), belief_constructor)
        agent_15.read(first_turn)
        plan_15 = agent_15.states[-1].plan

        if plan_15 is None:
            continue

        for i in range(len(subsets)):
            agent_10 = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4-0613")
            agent_10.example_subset = subsets[i]
            agent_10.feed_context(view.flatten().tolist(), belief_constructor)
            agent_10.read(first_turn)
            plan_10 = agent_10.states[-1].plan

            if plan_15 is not None and plan_10 is not None:
                accuracies[i].append((plan_15.dots == plan_10.dots).all())
            elif plan_15 is None and plan_10 is None:
                accuracies[i].append(True)
            else:
                accuracies[i].append(False)
        counter += 1
        print("COUNTER", counter)

    if counter >= 20:
        print(np.array(accuracies).mean(1).mean(), np.array(accuracies).mean(1).std())
        import pdb; pdb.set_trace()
