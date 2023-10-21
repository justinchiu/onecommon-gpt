from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai
import ast

from oc.gptagent.construct_prompt import BOARDS, OC_INSTRUCTIONS
from oc.gptagent.construct_prompt import verbalize_board, construct_prompt


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

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": OC_INSTRUCTIONS},
                {"role": "user", "content": prompt_prefix},
            ],
            stop="\n",
        )
        utterance = response.choices[0].message.content
        self.turns.append(utterance)
        split_utterance = utterance.split()
        if split_utterance[1] == "Select":
            selected_id = split_utterance[2]
            self.selected_dot = list(map(lambda x: x["id"], self.board)).index(selected_id)
            split_utterance = ["<selection>"]
        return split_utterance

    def choose(self):
        if self.selected_dot is None:
            prompt_prefix = construct_prompt(self.board_desc, self.turns, self.agent_id)
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": OC_INSTRUCTIONS},
                    {"role": "user", "content": prompt_prefix + "\nYou: Select"},
                ],
                stop="\n",
            )
            split_utterance = response.choices[0].message.content.split()
            selected_id = split_utterance[0]
            self.selected_dot = list(map(lambda x: x["id"], self.board)).index(selected_id)
        return self.selected_dot
