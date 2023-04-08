import numpy as np
from pathlib import Path
import sys
import minichain

from ocdata import get_data
from ocagent import Agent
from features import size_map5, color_map5, size_color_descriptions, process_ctx, render

# fried arguments

oc_dir = Path("../onecommon/aaai2020/experiments")
model_file = oc_dir / "/expts/rel3_tsel_ref_dial_model_separate/jc-baseline/baseline/1/1_best.th"
detector_file = oc_dir / "serialized_models/markable_detector_with_dict_1.th"

# load scripts and initialize...better to make everything into libraries
sys.path.append(str(oc_dir.resolve()))
from belief_agent import BeliefAgent
import utils
markable_detector = utils.load_model(detector_file, prefix_dir=None, map_location="cpu")
markable_detector.eval()

model = utils.load_model(model_file, prefix_dir=None, map_location="cpu")
import pdb; pdb.set_trace()
# on to generation

num_buckets = 3
#num_buckets = 5

run_example = "C_0dd19b44543141beb1737f391f2a1899"

data, _ = get_data()
data = [ex for ex in data if ex["chat_id"] == run_example]

example = data[0]

chatid = example["chat_id"]
scenarioid = example["scenario_id"]
print(scenarioid)
print(chatid)

view = example["context"]
turns = example["dialogue"]
referents = example["all_referents"]
dot_ids = example["real_ids"]

for t in range(len(turns)):
    text = turns[t]
    past = turns[:t]
    plan = referents[t]

    refs = [r["target"] for r in plan]
    planbool = np.array(refs).any(0)

    print("plan")
    print([id for present, id in zip(planbool, dot_ids) if present])

    size_color = process_ctx(view, num_size_buckets=num_buckets, num_color_buckets=num_buckets)
    dots = size_color[planbool]
    descs = size_color_descriptions(dots)
    xy = view[planbool,:2]

    descstring = []
    for (size, color), (x,y) in zip(descs, xy):
        descstring.append(f"* A {size} and {color} dot (x={x:.2f},y={y:.2f})")

    with minichain.start_chain("tmp.txt") as backend:
        agent = Agent(backend, "codegen", "templateonly", "gpt-3.5-turbo")
        out = agent.generate_text(plan, past, view)
        import pdb; pdb.set_trace()


