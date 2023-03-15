from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai

from prompt import HEADER, Understand, Execute, Generate


@dataclass
class State:
    memory: list[tuple[str, str]]
    human_input: str = ""

    def push(self, response: str) -> "State":
        memory = self.memory if len(self.memory) < MEMORY else self.memory[1:]
        return State(memory + [(self.human_input, response)])


class Agent:
    def __init__(self, backend):
        self.backend = backend
        self.understand = Understand(backend.OpenAI(
            model = "code-davinci-002",
            max_tokens=2048,
        ))
        self.execute = Execute(backend.Python())

    def read(self):
        pass

    def write(self):
        pass

    def resolve_reference(self, text, past, view):
        # ensure text ends in punctuation
        # codex seems to need a period
        if re.match('^[A-Z][^?!.]*[?.!]$', text) is None:
            text += "."

        kwargs = dict(header=HEADER, text=text, past=past, view=view)

        # print for debugging
        #input = self.understand.print(kwargs)
        #print(input)

        out = self.understand(kwargs)
        #print(out)

        # new input
        input = self.understand.print(dict(text=text, past=past, view=view))
        kw = dict(header=HEADER, code=input + out, dots=view.tolist())

        # debugging
        input = self.execute.print(kw)
        print(input)
        
        result = self.execute(kw)

        mention = np.zeros(7, dtype=bool)
        mention[result] = 1

        return mention, past + [(text.strip(), out.strip())]


    def plan(self, past, view):
        import pdb; pdb.set_trace()
        raise NotImplementedError


    def generate(self, past, view):
        kwargs = dict(plan=plan, past=past, view=view)
        self.understand.print(kwargs)
        out = self.understand(kwargs)
        return out
