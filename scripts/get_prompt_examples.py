from pathlib import Path
import re
from oc.dynamic_prompting.blocks import BLOCKS

strings = []
for block in BLOCKS:
    turn = block["turn"]
    text = block["text"]
    type = block["type"]
    if type != "No op.":
        state = block["state"]
        refturns = re.findall(r"\d+", state)
        olddots = block["dots"]
        newdots = block["newdots"]

        string = f"""Turn {turn}: {text}
Type: {type}
Reference turn: {refturns[0] if len(refturns) > 0 else None}
Dots: {olddots}
Newdots: {newdots}"""
    else:
        string = f"""Turn {turn}: {text}
Type: {type}"""
    print(string)
    strings.append(string)

with Path("scratch/short-example.txt").open("w") as f:
    f.write("\n\n".join(strings))
