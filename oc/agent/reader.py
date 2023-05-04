import numpy as np

from oc.prompt import HEADER, Understand, Execute, Reformat, Confirm
from oc.prompt import Parse, ParseUnderstand
from oc.prompt import UnderstandMc

from oc.prompt import UnderstandShort, ExecuteShort

class ReaderMixin:
    def __init__(self, backend, refres, gen, model):
        self.refres = refres

        self.reformat = Reformat(backend.OpenAIChat(
            model = "gpt-3.5-turbo",
            max_tokens = 128,
        ))

        self.confirm = Confirm(backend.OpenAIChat(
            model = "gpt-4",
            max_tokens = 5,
        ))

        if refres == "codegen":
            self.understand = Understand(backend.OpenAIChat(
                model = model,
                max_tokens=1024,
            ))
            self.execute = Execute(backend.Python())
        elif refres == "shortcodegen":
            self.understand = UnderstandShort(backend.OpenAIChat(
                model = model,
                max_tokens=64,
            ))
            self.execute = ExecuteShort(backend.Python())
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

        super(ReaderMixin, self).__init__(backend, refres, gen, model)

    def read(self, input_words): 
        # process input
        text = " ".join(input_words)
        past = self.past
        ctx = self.ctx
        preds, past, extra = self.resolve_reference(text, past, ctx)

        parsed_text = extra["parsedtext"]

        # confirmation / deny / none
        confirmation = self.confirm(dict(text=parsed_text))

        if len(self.plans) > 0:
            prev_plan = self.plans[-1]
            if confirmation is True:
                self.update_belief(prev_plan.dots, 1)
            elif confirmation is False:
                self.update_belief(prev_plan.dots, 0)
            elif confirmation is None:
                pass

        # if they asked a question and your answer is yes, update belief
        # your answer is yes if preds is not empty
        if preds.sum() > 0:
            self.update_belief(preds[0], 1)

        # TODO: wrap state update in function
        # TODO: management of past stack
        # should only keep around past if within line of questioning, eg same dots
        # update state??
        self.past = past
        self.preds.append(preds)
        self.plans.append(None)
        self.confirmations.append(confirmation)
        self.write_extras.append(None)
        self.read_extras.append(extra)


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

