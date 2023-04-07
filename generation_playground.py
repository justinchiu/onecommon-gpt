import numpy as np
import minichain

from ocdata import get_data
from agent import Agent
from features import size_map5, color_map5, size_color_descriptions, process_ctx, render

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
    plan = np.array(refs).any(0)

    print("plan")
    print([id for present, id in zip(plan, dot_ids) if present])

    size_color = process_ctx(view, num_size_buckets=num_buckets, num_color_buckets=num_buckets)
    dots = size_color[plan]
    descs = size_color_descriptions(dots)
    xy = view[plan,:2]

    descstring = []
    for (size, color), (x,y) in zip(descs, xy):
        descstring.append(f"* A {size} and {color} dot (x={x:.2f},y={y:.2f})")

    with minichain.start_chain("tmp.txt") as backend:
        agent = Agent(backend, "codegen", "scxy", "gpt-3.5-turbo")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        print("INPUT")
        print(agent.generate.print(kwargs))
        out = agent.generate(kwargs)
        print(out)
        import pdb; pdb.set_trace()
