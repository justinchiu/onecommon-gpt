
from enum import Enum
from pathlib import Path
import re
from oc.dynamic_prompting.blocks import BLOCKS
from oc.agent.utils import Qtypes


def question_type():
    strings = []
    for block in BLOCKS:
        turn = block["turn"]
        text = block["text"]
        type = block["type"]
        if type == "Follow up question, new dots.":
            state = block["state"]
            refturns = re.findall(r"\d+", state)
            olddots = block["dots"]
            newdots = block["newdots"]
            numnew = len(re.sub(",$", "", newdots).split(","))

            string = f"""Turn {turn}: {text}
Type: {type}
New dots: {numnew}"""
        elif type == "Follow up question, no new dots.":
            state = block["state"]
            refturns = re.findall(r"\d+", state)
            olddots = block["dots"]
            newdots = block["newdots"]

            string = f"""Turn {turn}: {text}
Type: {type}"""
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
        strings.append(string)
    return strings

def constraints():
    strings = []
    for block in BLOCKS:
        turn = block["turn"]
        text = block["text"]
        type = block["type"]
        state = block["state"]
        if type == Qtypes.NOOP.value:
            strings.append(f"Turn {turn}\nText: {text}\nType: {type}\nCode:\n```\npass\n```")
        elif type == Qtypes.START.value or type == Qtypes.FOLD.value:
            dots = block["dots"]
            constraints = block["constraints"]
            string = f"Turn {turn}\nText: {text}\nType: {type}\nDots: {dots}\nCode:"
            constraint_string = "\n".join(f"{x['name']} = {x['code']}" for x in constraints)
            strings.append("\n".join([string, "```", constraint_string, "```"]))
        else:
            dots = block["dots"]
            refturns = re.findall(r"\d+", state)
            constraints = block["constraints"]
            string = f"Turn {turn}\nText: {text}\nType: {type}\nDots: {dots}\nPrevious turn: {refturns[0]}\nCode:"
            constraint_string = "\n".join(f"{x['name']} = {x['code']}" for x in constraints)
            strings.append("\n".join([string, "```", constraint_string, "```"]))

    return strings

def constraints_no_var():
    strings = []
    for block in BLOCKS:
        turn = block["turn"]
        text = block["text"]
        type = block["type"]
        state = block["state"]
        if type == Qtypes.NOOP.value:
            strings.append(f"Turn {turn}\nText: {text}\nType: {type}\nCode:\n```\npass\n```")
        elif type == Qtypes.START.value or type == Qtypes.FOLD.value:
            dots = block["dots"]
            constraints = block["constraints"]
            string = f"Turn {turn}\nText: {text}\nType: {type}\nDots: {dots}\nCode:"
            constraint_string = "\n".join(f"{x['code']}" for x in constraints)
            strings.append("\n".join([string, "```", constraint_string, "```"]))
        else:
            dots = block["dots"]
            refturns = re.findall(r"\d+", state)
            constraints = block["constraints"]
            string = f"Turn {turn}\nText: {text}\nType: {type}\nDots: {dots}\nPrevious turn: {refturns[0]}\nCode:"
            constraint_string = "\n".join(f"{x['code']}" for x in constraints)
            strings.append("\n".join([string, "```", constraint_string, "```"]))

    return strings

