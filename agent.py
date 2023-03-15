from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai

from prompt import HEADER, Understand, Execute, Generate
from template import size_map5, color_map5, size_color_descriptions, process_ctx


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
        self.generate = Generate(backend.OpenAI(
            model = "text-davinci-003",
            max_tokens=512,
        ))

    def read(self):
        pass

    def write(self):
        pass

    def resolve_reference(self, text, past, view, info=None):
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

        # new input for python execution
        input = self.understand.print(dict(text=text, past=past, view=view))
        kw = dict(info=info, header=HEADER, code=input + out, dots=view.tolist())

        # debugging
        input = self.execute.print(kw)
        print(input)
        
        result = self.execute(kw)
        print(result)

        mention = np.zeros(7, dtype=bool)
        mention[result] = 1

        return mention, past + [(text.strip(), out.strip())]


    def plan(self, past, view, info=None):
        import pdb; pdb.set_trace()
        raise NotImplementedError


    def generate_text(self, plan, past, view, info=None):
        # process plan
        refs = [r["target"] for r in plan]
        size_color = process_ctx(view)
        dots = size_color[np.array(refs).any(0)]
        descs = size_color_descriptions(dots)
        descstring = []
        for size, color in descs:
            descstring.append(f"* A {size} and {color} dot")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        print("INPUT")
        print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out]
