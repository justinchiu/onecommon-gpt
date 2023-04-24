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

from oc.agent.reader import ReaderMixin
from oc.agent.planner import PlannerMixin
from oc.agent.writer import WriterMixin

class Agent(ReaderMixin, PlannerMixin, WriterMixin):
    def __init__(self, backend, refres, gen, model="gpt-3.5-turbo"):
        # number of buckets for size and color
        self.num_buckets = 3
        self.backend = backend
        self.model = model

        super().__init__(backend, refres, gen, model)

    # necessary functions for onecommon
    def feed_context(self, ctx):
        self.ctx = ctx
        self.past = []
        self.belief = CostBelief(
            7, ctx,
            absolute = True,
            num_size_buckets = self.num_buckets,
            num_color_buckets = self.num_buckets,
            use_diameter = False,
            use_contiguity = False,
        )


    def choose(self):
        import pdb; pdb.set_trace()
        pass



    # PLANNING
    def plan(self, past, view, info=None):
        import pdb; pdb.set_trace()
        raise NotImplementedError

