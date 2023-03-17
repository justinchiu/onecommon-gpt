import numpy as np
import pdb
import features
from itertools import combinations

from ocdata import get_data
from features import render, size_color_descriptions
from code.shapes import is_triangle, is_line, is_contiguous
from code.spatial import is_close

def get_generation_feature_prompt_examples():
    train, valid = get_data()

    for example in train:
        context = example["context"]
        turns = example["dialogue"]
        refs = example["all_referents"]

        xy = context[:,:2]
        sc = features.process_ctx(context)

        print(example["scenario_id"])
        print(example["chat_id"])

        # print for multiple choice GPT resolution
        print("Description:")
        for i, ((s, c), (x, y)) in enumerate(zip(size_color_descriptions(sc), xy)):
            print(f"* Dot {i+1}: {s} and {c} (x={x:.2f}, y={y:.2f})")

        for t in range(len(turns)):
            plan = np.array([r["target"] for r in refs[t]]).any(0)
            turn = turns[t]

            print(turn)

            mentioned_dots = (plan.nonzero()[0] + 1).tolist()
            if len(mentioned_dots) > 0:
                print(f"Mentions dots: {mentioned_dots}")
            else:
                print(f"Mentions dots: None")


            """
            if plan.sum() <= 0:
                continue
            # print stuff out for template prompt
            desc = render(plan, context)
            print("PAST")
            print(turns[:t])
            print("DESC")
            print(desc)
            print("TURN")
            print(turn)
            pdb.set_trace()
            """
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    get_generation_feature_prompt_examples()
