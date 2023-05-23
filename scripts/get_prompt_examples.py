from pathlib import Path
import re
from oc.dynamic_prompting.blocks import BLOCKS

strings = []
for block in BLOCKS:
    turn = block["turn"]
    text = block["text"]
    type = block["type"]
    if type == "Follow up question, new dot.":
        state = block["state"]
        refturns = re.findall(r"\d+", state)
        olddots = block["dots"]
        newdots = block["newdots"]
        numnew = len(re.sub(",$", "", newdots).split(","))

        string = f"""Turn {turn}: {text}
Type: Follow up question, new dots.
Newdots: {numnew}"""
    elif type == "Follow up question.":
        state = block["state"]
        refturns = re.findall(r"\d+", state)
        olddots = block["dots"]
        newdots = block["newdots"]

        string = f"""Turn {turn}: {text}
Type: Follow up question, no new dots."""
    elif type == "New question.":
        state = block["state"]

        string = f"""Turn {turn}: {text}
Type: {type}"""
    elif type == "Select a dot.":
        state = block["state"]
        refturns = re.findall(r"\d+", state)
        string = f"""Turn {turn}: {text}
Type: {type}"""
    elif type == "No op.":
        string = f"""Turn {turn}: {text}
Type: {type}"""
    print(string)
    strings.append(string)

with Path("scratch/short-example.txt").open("w") as f:
    f.write("\n\n".join(strings))
