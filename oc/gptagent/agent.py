from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai
import ast
import string
import tenacity

from oc.gptagent.construct_prompt import BOARDS, OC_INSTRUCTIONS
from oc.gptagent.construct_prompt import verbalize_board, construct_prompt


@tenacity.retry#(wait=tenacity.wait_fixed(0.5))
def complete(model, system, user):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stop="\n",
    )
    return response

class Agent:
    def __init__(self, backend, refres, gen, model="gpt-3.5-turbo"):
        # number of buckets for size and color
        self.num_buckets = 3
        self.belief_threshold = 0.8
        self.backend = backend
        self.model = model

    # necessary functions for onecommon
    def feed_context(
        self,
        ctx,
        flip_y=False, belief_constructor=None,
        scenario_id=None, agent_id=0,
    ):
        self.scenario_id = scenario_id
        self.agent_id = agent_id
        # CHECK IF WE NEED TO FLIP Y-AXIS
        board = BOARDS[scenario_id]["kbs"][agent_id]
        self.board = board
        self.board_desc = verbalize_board(board)

        # dialogue state
        self.turns = []
        self.selected_dot = None


    def read(self, input_words):
        self.turns.append(" ".join(input_words))

    def write(self):
        prompt_prefix = construct_prompt(self.board_desc, self.turns, self.agent_id)

        response = complete(self.model, OC_INSTRUCTIONS, prompt_prefix)
        #import pdb; pdb.set_trace()
        utterance = response.choices[0].message.content
        self.turns.append(utterance)
        split_utterance = utterance.split()
        if split_utterance[1] == "Select":
            selected_id = split_utterance[2].translate(str.maketrans('', '', string.punctuation))
            string_ids = list(map(lambda x: x["id"], self.board))
            self.selected_dot = (
                string_ids.index(selected_id)
                if selected_id in string_ids
                else 0
            )
            split_utterance = ["<selection>"]
        return split_utterance

    def choose(self):
        if self.selected_dot is None:
            prompt_prefix = construct_prompt(self.board_desc, self.turns, self.agent_id)
            response = complete(self.model, OC_INSTRUCTIONS, prompt_prefix + "\nYou: Select")
            split_utterance = response.choices[0].message.content.split()
            selected_id = split_utterance[0].translate(str.maketrans('', '', string.punctuation))
            string_ids = list(map(lambda x: x["id"], self.board))
            self.selected_dot = (
                string_ids.index(selected_id)
                if selected_id in string_ids
                else 0
            )

        return self.selected_dot
