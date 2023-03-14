from dataclasses import dataclass
import numpy as np
from pathlib import Path
import openai

from prompt import Understand, Execute, Generate


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
            max_tokens=256,
        ))
        self.execute = Execute(backend.Python())

    def read(self):
        pass

    def write(self):
        pass

    def resolve_reference(self, text, past, view):
        kwargs = dict(text=text, past=past, view=view)
        input = self.understand.print(kwargs)
        print(input)
        out = self.understand(kwargs)
        print(out)
        kw = dict(code=input + out)
        print(self.execute.print(kw))
        with Path("temporary_code.py").open("w") as f:
            f.write(self.execute.print(kw))
        import temporary_code
        import pdb; pdb.set_trace()
        result = self.execute(kw)
        import pdb; pdb.set_trace()
        return np.zeros(7, dtype=bool)

    def plan(self, past, view):
        import pdb; pdb.set_trace()
        raise NotImplementedError


    def generate(self, past, view):
        kwargs = dict(plan=plan, past=past, view=view)
        self.understand.print(kwargs)
        out = self.understand(kwargs)
        return out
