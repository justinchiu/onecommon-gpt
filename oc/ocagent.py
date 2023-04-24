from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import openai
import ast

from oc.gen.features import size_map5, color_map5, size_color_descriptions, process_ctx, render

from oc.prompt import HEADER, Understand, Execute, Generate, Reformat
from oc.prompt import Parse, ParseUnderstand
from oc.prompt import GenerateScxy, GenerateTemplate
from oc.prompt import UnderstandMc


class Agent:
    def __init__(self, backend, refres, gen, model="gpt-3.5-turbo"):
        self.backend = backend

        self.refres = refres
        self.gen = gen

        self.model = model

        self.reformat = Reformat(backend.OpenAIChat(
            model = "gpt-3.5-turbo",
            max_tokens = 128,
        ))

        if refres == "codegen":
            self.understand = Understand(backend.OpenAIChat(
                model = model,
                max_tokens=1024,
            ))
            self.execute = Execute(backend.Python())
        elif refres == "parsecodegen":
            self.parse = Parse(backend.OpenAIChat(
                model = model,
                max_tokens = 512,
            ))
            print(f"RUNNING UNDERSTANDING WITH MODEL: {model}")
            self.understand = ParseUnderstand(backend.OpenAIChat(
                model = model,
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

    # necessary functions for onecommon
    def feed_context(self, ctx):
        self.ctx = ctx
        self.past = []

    def read(self, input_words):
        # process input
        text = " ".join(input_words)
        past = self.past
        ctx = self.ctx
        preds, past, extra = self.resolve_reference(text, past, ctx)
        # TODO: management of past stack
        # should only keep around past if within line of questioning, eg same dots
        # update state
        self.past = past

    def write(self):
        pass

    def choose(self):
        pass


    # helper functions
    def reformat_text(self, text, usespeaker=True):
        speaker = "You" if "You:" in text else "Them"
        utt = text.replace("You: ", "").replace("Them: ", "")
        #print(self.reformat.print(dict(source=utt.strip())))
        out = self.reformat(dict(source=utt)).strip()
        #print("Reformatted")
        #print(text)
        text = f"{speaker}: {out}" if usespeaker else out
        #print(text)
        return text

    # RESOLUTION
    def resolve_reference(self, text, past, view, info=None):
        # dispatch
        if self.refres == "codegen":
            return self.resolve_reference_codegen(text, past, view, info=info)
        elif self.refres == "parsecodegen":
            return self.resolve_reference_parse_codegen(text, past, view, info=info)
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

        return mention, past + [(text.strip(), f"Mentions dots: {out.strip()}")], None

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

        return mentions, past + [(text.strip(), f"def {out.strip()}")], {
            "parsedtext": text,
        }

    def resolve_reference_parse_codegen(self, text, past, view, info=None):
        text = self.reformat_text(text, usespeaker=False)

        parse_prompt = self.parse.print(dict(text=text))
        #print(parse_prompt)
        parsed, confirmation, desc, selection = self.parse(dict(text=text))
        #print(confirmation)
        #print(desc)

        understand_input_text = f"Confirmation: {parsed}"

        kwargs = dict(header=HEADER, text=understand_input_text, past=past, view=view)

        codeblock = self.understand(kwargs)

        # new input for python execution
        input = self.understand.print(dict(text=understand_input_text, past=past, view=view))
        kw = dict(info=info, header=HEADER, code=input + codeblock, dots=view.tolist())

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

        return (
            mentions,
            past + [(understand_input_text.strip(), f"def {codeblock.strip()}")],
            {
                "parsedtext": text,
                "bullet": understand_input_text,
            },
        )


    # PLANNING
    def plan(self, past, view, info=None):
        import pdb; pdb.set_trace()
        raise NotImplementedError


    # GENERATION
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
        return out, past + [out], None

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
        return out, past + [out], None

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
        return out, past + [out], None

    def generate_text_template_only(self, plan, past, view, info=None):
        if len(plan) == 0:
            # no references...
            return "okay", past + ["okay"]

        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)
        desc = render(plan, view, num_buckets=3)
        print(desc)
        return desc, past + [desc], None

    def generate_new_config(self, plan, past, view, info=None):
        pass

    def generate_followup(self, plan, past, view, info=None):
        pass

    def generate_select(self, pan, past, view, info=None):
        pass