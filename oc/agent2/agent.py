from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai
import ast

from oc.prompt import HEADER, Understand, Execute, Generate, Reformat
from oc.prompt import Parse, ParseUnderstand
from oc.prompt import GenerateScxy, GenerateTemplate
from oc.prompt import UnderstandMc

from oc.agent2.utils import State, Past
from oc.agent2.reader import ReaderMixin
from oc.agent2.planner import PlannerMixin
from oc.agent2.writer import WriterMixin

from oc.belief.belief import CostBelief, OrBelief, PriorType


class Agent(ReaderMixin, PlannerMixin, WriterMixin):
    def __init__(self, backend, refres, gen, model="gpt-3.5-turbo"):
        # number of buckets for size and color
        self.num_buckets = 3
        self.belief_threshold = 0.8
        self.backend = backend
        self.model = model

        super().__init__(backend, refres, gen, model)

    # necessary functions for onecommon
    def feed_context(self, ctx, flip_y=False, belief_constructor=None):
        self.ctx = np.array(ctx, dtype=float).reshape((7,4))
        if flip_y:
            self.ctx[:,1] = -self.ctx[:,1]

        #self.belief = CostBelief(
        self.belief = OrBelief(
            7, ctx,
            absolute = True,
            num_size_buckets = self.num_buckets,
            num_color_buckets = self.num_buckets,
            use_diameter = False,
            use_contiguity = False,
            prior_type = PriorType.ISING,
        )

        # dialogue state
        self.states = [State(
            belief_dist = self.belief.prior,
            plan = None,
            speaker = None,
            turn = -1,
            past = Past(
                classify_past = [],
                understand_past = [],
                execute_past = [],
            ),
            text = None,
        )]

