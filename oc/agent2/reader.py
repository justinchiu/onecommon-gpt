import numpy as np
import pprint

from oc.prompt import HEADER, Understand, Execute, Reformat, Confirm
from oc.prompt import Parse, ParseUnderstand
from oc.prompt import UnderstandMc

from oc.prompt import UnderstandShort, ExecuteShort
from oc.prompt import UnderstandShort2, ExecuteShort2
from oc.prompt import Classify

from oc.dynamic_prompting.blocks import BLOCKS
from oc.dynamic_prompting.construct import question_type, constraints_dots

from oc.agent2.utils import Speaker, Plan, State, Qtypes

from oc.belief.belief_utils import get_config_idx


letters = "abcdefghijklmnop"

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
            self.classify = Classify(backend.OpenAIChat(
                model = model,
                max_tokens = 64,
            ))
            self.understand = UnderstandShort2(backend.OpenAIChat(
                model = model,
                max_tokens=256,
            ))
            self.execute = ExecuteShort2(backend.Python())
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
            all_dots = preds if preds is not None and confirmed else None,
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
        import time
        read_start_time = time.perf_counter()

        speaker = "You" if "You:" in text else "Them"
        text = self.reformat_text(text, usespeaker=False)

        classify_blocks = question_type()

        start_time = time.perf_counter()
        qtype, num_new_dots, classify_past = self.classify(dict(
            blocks=classify_blocks,
            past=past.classify_past,
            text=text,
        ))
        print(f"Classify: {time.perf_counter() - start_time} seconds")

        num_prev_dots = 0
        prev_dots = None
        if qtype == Qtypes.FNEW.value or qtype == Qtypes.FOLD.value:
            previous_dots, last_turn = self.get_last_confirmed(self.states).sum()
            num_prev_dots = previous_dots.sum().item()
            prev_dots = ",".join(letters[:num_prev_dots])
        dots = ",".join(letters[num_prev_dots:num_prev_dots+num_new_dots])

        understand_blocks = constraints_dots()
        understand_kwargs = dict(
            header = HEADER,
            blocks = understand_blocks,
            speaker = speaker,
            text = text,
            type = qtype,
            prev_dots = prev_dots,
            dots = dots,
        )
        understand_prompt = self.understand.print(understand_kwargs)
        print(understand_prompt)

        import time
        start_time = time.perf_counter()
        constraints = self.understand(understand_kwargs)
        print(f"Understand: {time.perf_counter() - start_time} seconds")
        print(f"Read until code: {time.perf_counter() - read_start_time} seconds")

        import pdb; pdb.set_trace()

        codeblock_dict = None
        if constraints is None:
            codeblock_dict = dict(
                noop = True,
                speaker = speaker,
                text = text,
                state = None,
            )
        elif qtype == Qtypes.NOOP.value:
            # construct codeblocks
            codeblock_dict = dict(
                noop = False,
                constraints = None,
                configs = None,
                dots = None,
                newconfigs = None,
                newdots = None,
                select = "False",
                speaker = speaker,
                text = text,
                state = "None",
            )
        elif qtype == Qtypes.START.value:
            # construct codeblocks
            codeblock_dict = dict(
                noop = False,
                constraints = constraints,
                configs = f"getsets(idxs, {num_new_dots})",
                dots = dots,
                newconfigs = "[0]",
                newdots = "_",
                select = "False",
                speaker = speaker,
                text = text,
                savedots = dots,
            )
        elif qtype == Qtypes.FOLD.value:
            # construct codeblocks
            codeblock_dict = dict(
                noop = False,
                constraints = constraints,
                configs = "state",
                dots = prev_dots,
                newconfigs = "[0]",
                newdots = "_",
                select = "False",
                speaker = speaker,
                text = text,
                savedots = prev_dots,
            )
        elif qtype == Qtypes.FNEW.value:
            # construct codeblocks
            codeblock_dict = dict(
                noop = False,
                constraints = constraints,
                configs = "state",
                dots = prev_dots,
                newconfigs = f"get{num_new_dots}idxs(idxs, exclude=[{prev_dots}])",
                newdots = dots,
                select = "False",
                speaker = speaker,
                text = text,
                savedots = ",".join([prev_dots, dots]),
            )
        elif qtype == Qtypes.SELECT.value:
            # construct codeblocks
            codeblock_dict = dict(
                noop = False,
                constraints = constraints,
                configs = "state",
                dots = dots,
                newconfigs = "[0]",
                newdots = "_",
                select = "True",
                speaker = speaker,
                text = text,
                savedots = "a",
            )

        # new input for python execution
        kw = dict(
            info=info,
            header=HEADER,
            blocks=past.execute_past + [codeblock_dict],
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
            Past(
                classify_past = classify_past,
                understand_past = [],
                execute_past = [],
            ),
            {
                "parsedtext": text,
                "speaker": speaker,
            },
        )
