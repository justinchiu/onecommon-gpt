import os
import openai
import json
import tiktoken

from oc.dot import Dot

json_file = f"{os.environ['HOME']}/research/onecommon/webapp/pomdp_experiments/all_chats.json"
with open(json_file, "r") as f:
    dialogues = json.load(f)

board_file = f'{os.environ["HOME"]}/research/onecommon/aaai2020/experiments/data/onecommon/shared_4.json'
with open(board_file, "r") as f:
    scenario_list = json.load(f)
boards = {
    scenario['uuid']: scenario
    for scenario in scenario_list
}

dialogue = dialogues[2]
scenario_id = dialogue["scenario_id"]
board = boards[scenario_id]["kbs"][0]

def verbalize_board(board):
    # `board` should be raw json dict from shared_4.json.
    # hopefully that's what's given to process_ctx?
    dots = [Dot(dot) for dot in board]
    return "\n".join(
        f"* Dot {d.id}: Position: ({d.x}, {d.y}) Radius: {d.size} Color: {d.color}"
        for i, d in enumerate(dots)
    )

board_desc = verbalize_board(board)

instructions = """You and your partner are trying to find one dot in common.
You both see overlapping but different view of a game board.
Your view contains 7 dots, a few of which are shared with your partner.
Your goal is to discuss groups of dots in order to arrive at a single shared dot."""

def convert_id(id):
    return "You" if id == 0 else "Them"

turns = [(t["agent"], t["data"]) for t in dialogue["events"] if t["action"] == "message"]
selects = [(t["agent"], int(t["data"].replace('"', ""))) for t in dialogue["events"] if t["action"] == "select"]

turn_string = "\n".join([f"{convert_id(id)}: {turn}" for id, turn in turns])
select_string = "\n".join([f"{convert_id(id)}: Select {turn}" for id, turn in selects])

prompt = f"""{instructions}

Dialogue 1
{verbalize_board(board)}

{turn_string}
{select_string}

Dialogue 2
{this_turn_string}
You:"""

print(prompt)

enc = tiktoken.encoding_for_model("gpt-4")
encoded_prompt = enc.encode(prompt)
print(len(encoded_prompt))
import pdb; pdb.set_trace()
