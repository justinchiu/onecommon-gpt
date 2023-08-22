from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai
import ast

from oc.gptagent.construct_prompt import BOARDS
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
        self.board_desc = verbalize_board(board)

        # dialogue state
        self.turns = []


    def read(self, input_words):
        self.turns.append(input_words)
        pass


    def write(self):
        prompt_prefix = construct_prompt(self.board_desc, self.turns, self.agent_id)

        import pdb; pdb.set_trace()
        out = "blah"
        self.turns.append(out)
        return out

