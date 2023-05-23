import numpy as np
import pprint

from oc.prompt import HEADER, Understand, Execute, Reformat, Confirm
from oc.prompt import Parse, ParseUnderstand
from oc.prompt import UnderstandMc

from oc.prompt import UnderstandShort, ExecuteShort
from oc.prompt import UnderstandShort2, ExecuteShort2
from oc.prompt import UnderstandJson, ExecuteJson

from oc.dynamic_prompting.blocks import BLOCKS

from oc.agent2.utils import Speaker, Plan, State

from oc.belief.belief_utils import get_config_idx


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

        if refres == "shortcodegen2":
            self.understand = UnderstandShort(backend.OpenAIChat(
                model = model,
                max_tokens=1024,
            ))
            self.execute = ExecuteShort(backend.Python())
        else:
            raise ValueError

        super(ReaderMixin, self).__init__(backend, refres, gen, model)

    def read(self, input_words):
        state = self.states[-1]

        print("IN READ")
        # process input
        text = " ".join(input_words)
        past = self.states[-1].past
        ctx = self.ctx
        preds, past, extra = self.resolve_reference(text, past, ctx)
        # maybe should rank preds by probability?

        parsed_text = extra["parsedtext"]

        # process our previous plan + their confirmation

        # confirmation / deny / none
        confirmation = self.confirm(dict(text=parsed_text))

        # update belief
        belief_dist = state.belief_dist
        updated_belief_dist = belief_dist
        if len(self.states) > 1 and state.plan is not None:
            prev_plan = state.plan
            if confirmation is True:
                updated_belief_dist = self.update_belief(belief_dist, prev_plan.dots, 1)
                print("UPDATED BELIEF confirmed")
                state.plan.confirmed = True
            elif confirmation is False:
                updated_belief_dist = self.update_belief(belief_dist, prev_plan.dots, 0)
                print("UPDATED BELIEF denied")
                state.plan.confirmed = False
            elif confirmation is None:
                updated_belief_dist = belief_dist

        # construct plan for what they said
        feats = None
        plan_idxs = None
        config_idx = None
        confirmed = None
        if preds is not None:
            # None => they didnt mention anything
            confirmed = preds.sum() > 0
            # sum == 0 => we dont see their plan
            if confirmed:
                feats = self.belief.get_feats(preds[0]) # pull out into fn
                plan_idxs = self.belief.resolve_utt(*feats) # pull out into fn
                config_idx = get_config_idx(preds[0], self.belief.configs)
        plan = Plan(
            dots = preds[0] if preds is not None and confirmed else None,
            config_idx = config_idx,
            feats = feats,
            plan_idxs = plan_idxs,
            confirmation = confirmation,
            confirmed = confirmed,
        )

        # if they asked a question and your answer is yes, update belief
        # your answer is yes if preds is not empty
        last_belief_dist = updated_belief_dist
        if plan.confirmed:
            last_belief_dist = self.update_belief(updated_belief_dist, preds[0], 1)
            print("UPDATED BELIEF we see")

        self.states.append(State(
            belief_dist = last_belief_dist,
            plan = plan,
            past = past,
            speaker = Speaker.THEM,
            turn = state.turn+1,
            read_extra = extra,
        ))

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
        if self.refres == "shortcodegen2":
            return self.resolve_reference_short_codegen2(text, past, view, info=info)
        else:
            raise ValueError

    def resolve_reference_short_codegen2(self, text, past, view, info=None):
        speaker = "You" if "You:" in text else "Them"
        text = self.reformat_text(text, usespeaker=False)

        kwargs = dict(
            header=HEADER,
            blocks = BLOCKS,
            speaker = speaker,
            text=text,
            past=past,
            view=view,
        )

        understand_prompt = self.understand.print(kwargs)
        print(understand_prompt)

        codeblock = self.understand(kwargs)

        codeblock_dict = None
        if codeblock is None:
            codeblock_dict = dict(
                noop = True,
                speaker = speaker,
                text = text,
                state = None,
            )
        else:
            codeblock_dict = dict(
                noop = False,
                code = codeblock.code,
                constraints = codeblock.constraints,
                dots = codeblock.dots,
                select = codeblock.select,
                speaker = codeblock.speaker,
                text = codeblock.text,
                state = codeblock.state,
            )

        # new input for python execution
        kw = dict(
            info=info,
            header=HEADER,
            blocks=past + [codeblock_dict],
            dots=view.tolist(),
        )

        # debugging execution input
        input = self.execute.print(kw)
        print(input)

        result = self.execute(kw)
        print(result)

        mentions = None
        if result is not None:
            num_preds = len(result)
            mentions = np.zeros((num_preds, 7), dtype=bool)
            for i in range(num_preds):
                mentions[i, result[i]] = 1

        return (
            mentions,
            past + [codeblock_dict],
            {
                "parsedtext": text,
                "speaker": speaker,
            },
        )
