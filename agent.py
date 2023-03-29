from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai
import ast

from features import size_map5, color_map5, size_color_descriptions, process_ctx, render

from prompt import HEADER, Understand, Execute, Generate, Reformat, Parse
from prompt import GenerateScxy, GenerateTemplate
from prompt import UnderstandMc


@dataclass
class State:
    memory: list[tuple[str, str]]
    human_input: str = ""

    def push(self, response: str) -> "State":
        memory = self.memory if len(self.memory) < MEMORY else self.memory[1:]
        return State(memory + [(self.human_input, response)])


class Agent:
    def __init__(self, backend, refres, gen, parse):
        self.backend = backend

        self.refres = refres
        self.gen = gen
        self.parse = parse

        self.reformat = Reformat(backend.OpenAIChat(
            model = "gpt-3.5-turbo",
        #self.reformat = Reformat(backend.OpenAI(
        #    model = "text-davinci-003",
            max_tokens = 128,
        ))

        if refres == "codegen":
            #self.understand = Understand(backend.OpenAI(
            #    model = "code-davinci-002",
            self.understand = Understand(backend.OpenAIChat(
                model = "gpt-3.5-turbo",
                #model = "gpt-4",
                max_tokens=1024,
            ))
            self.execute = Execute(backend.Python())
        elif refres == "parsecodegen":
            self.parse = Parse(backend.OpenAIChat(
                model = "gpt-3.5-turbo",
                max_tokens = 512,
            ))
            self.understand = ParseUnderstand(backend.OpenAIChat(
                model = "gpt-3.5-turbo",
                #model = "gpt-4",
                max_tokens=1024,
            ))
            self.execute = Execute(backend.Python())
        elif refres == "mc":
            self.understand = UnderstandMc(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=1024,
            ))
        else:
            raise ValueError

        if gen == "sc":
            self.generate = Generate(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=512,
            ))
        elif gen == "scxy":
            self.generate = GenerateScxy(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=512,
            ))
        elif gen == "template":
            self.generate = GenerateTemplate(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=1024,
            ))
        elif gen == "templateonly":
            pass
        else:
            raise ValueError


    def read(self):
        pass

    def write(self):
        pass

    def reformat_text(self, text):
        speaker = "You" if "You:" in text else "Them"
        utt = text.replace("You: ", "").replace("Them: ", "")
        #print(self.reformat.print(dict(source=utt.strip())))
        out = self.reformat(dict(source=utt)).strip()
        #print("Reformatted")
        #print(text)
        text = f"{speaker}: {out}"
        #print(text)
        return text

    def resolve_reference(self, text, past, view, info=None):
        # dispatch
        if self.refres == "codegen":
            return self.resolve_reference_codegen(text, past, view, info=info)
        elif self.refres == "mc":
            return self.resolve_reference_mc(text, past, view, info=info)
        else:
            raise ValueError

    def resolve_reference_mc(self, text, past, view, info=None):
        xy = view[:,:2]
        sc = process_ctx(view)

        # print for multiple choice GPT resolution
        viewstr = []
        for i, ((s, c), (x, y)) in enumerate(zip(size_color_descriptions(sc), xy)):
            viewstr.append(f"* Dot {i+1}: {s} and {c} (x={x:.2f}, y={y:.2f})")
        view = "\n".join(viewstr)

        kwargs = dict(text=text, past=past, view=view)
        print(self.understand.print(kwargs))
        out = self.understand(kwargs)

        result = ast.literal_eval(out)
        print("PRED")
        mention = np.zeros(7, dtype=bool)
        if result is not None:
            result = np.array(result) - 1
            print(result)
            mention[result] = 1

        return mention, past + [(text.strip(), f"Mentions dots: {out.strip()}")]

    def resolve_reference_codegen(self, text, past, view, info=None):
        text = self.reformat_text(text)

        kwargs = dict(header=HEADER, text=text, past=past, view=view)

        out = self.understand(kwargs)

        # new input for python execution
        input = self.understand.print(dict(text=text, past=past, view=view))
        kw = dict(info=info, header=HEADER, code=input + out, dots=view.tolist())

        # debugging
        input = self.execute.print(kw)
        print(input)
        
        result = self.execute(kw)
        print(result)
        if result is None:
            result = []

        num_preds = len(result)
        mentions = np.zeros((num_preds, 7), dtype=bool)
        for i in range(num_preds):
            mentions[i, result[i]] = 1

        return mentions, past + [(text.strip(), f"def {out.strip()}")]


    def plan(self, past, view, info=None):
        import pdb; pdb.set_trace()
        raise NotImplementedError

    def generate_text(self, plan, past, view, info=None):
        if self.gen == "sc":
            return self.generate_text_sc(plan, past, view, info)
        if self.gen == "scxy":
            return self.generate_text_scxy(plan, past, view, info)
        elif self.gen == "template":
            return self.generate_text_template(plan, past, view, info)
        elif self.gen == "templateonly":
            return self.generate_text_template_only(plan, past, view, info)
        else:
            raise ValueError

    def generate_text_sc(self, plan, past, view, info=None):
        # process plan
        refs = [r["target"] for r in plan]
        size_color = process_ctx(view)
        dots = size_color[np.array(refs).any(0)]
        descs = size_color_descriptions(dots)
        descstring = []
        for size, color in descs:
            descstring.append(f"* A {size} and {color} dot")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        #print("INPUT")
        #print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out]

    def generate_text_scxy(self, plan, past, view, info=None):
        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)

        size_color = process_ctx(view)
        dots = size_color[plan]
        descs = size_color_descriptions(dots)
        xy = view[plan,:2]


        descstring = []
        for (size, color), (x,y) in zip(descs, xy):
            descstring.append(f"* A {size} and {color} dot (x={x:.2f},y={y:.2f})")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        print("INPUT")
        print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out]

    def generate_text_template(self, plan, past, view, info=None):
        if len(plan) == 0:
            # no references...
            return "okay", past + ["okay"]

        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)
        desc = render(plan, view)

        kwargs = dict(plan=desc, past="\n".join(past))
        #print("INPUT")
        #print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out]

    def generate_text_template_only(self, plan, past, view, info=None):
        if len(plan) == 0:
            # no references...
            return "okay", past + ["okay"]

        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)
        desc = render(plan, view)
        print(desc)
        return desc, past + [desc]

