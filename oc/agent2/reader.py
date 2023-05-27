import numpy as np
import pprint

from oc.prompt import HEADER, Understand, Execute, Reformat, Confirm
from oc.prompt import Parse, ParseUnderstand
from oc.prompt import UnderstandMc

from oc.prompt import UnderstandShort, ExecuteShort
from oc.prompt import UnderstandShort2, ExecuteShort2
from oc.prompt import Classify, ClassifyZeroshot

from oc.dynamic_prompting.blocks import BLOCKS
from oc.dynamic_prompting.construct import question_type, constraints_dots

from oc.agent2.utils import Speaker, Plan, State, Qtypes, Past
from oc.agent2.utils import StartPlan, FollowupPlan, SelectPlan

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
            #self.classify = ClassifyZeroshot(backend.OpenAIChat(
                model = model,
                max_tokens = 16,
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

        parsed_text = self.reformat_text(text, usespeaker=False)
        # confirmation / deny / none
        confirmation = self.confirm(dict(text=f"Them: {parsed_text}"))

        # process our previous plan + their confirmation
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
                print("NO UPDATED BELIEF none")
                updated_belief_dist = belief_dist

        # maybe should rank preds by probability?
        plan, past, extra = self.resolve_reference(parsed_text, past, ctx)

        # if they asked a question and your answer is yes, update belief
        # your answer is yes if preds is not empty
        last_belief_dist = updated_belief_dist
        if plan is not None and plan.confirmed:
            last_belief_dist = self.update_belief(updated_belief_dist, plan.dots, 1)
            print("UPDATED BELIEF we see")

        self.states.append(State(
            belief_dist = last_belief_dist,
            plan = plan,
            past = past,
            speaker = Speaker.THEM,
            turn = state.turn+1,
            read_extra = {"parsed_text": parsed_text},
            text = text,
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
        speaker = "Them"
        import time
        read_start_time = time.perf_counter()

        #"""
        # For few-shot question classification
        classify_blocks = question_type()
        classify_kwargs = dict(
            blocks=classify_blocks,
            past = [
                f"{state.text}\n"
                f"Type: {state.plan.qtype.value if state.plan is not None else 'Qtypes.NOOP.value'}\n"
                f"New dots: {state.plan.new_dots if state.plan is not None else 0}"
                for state in self.states if state.turn >= 0
            ],
            text=text,
        )
        """
        classify_kwargs = dict(
            past = [state.text for state in self.states if state.turn >= 0],
            text = text,
        )
        """
        print(self.classify.print(classify_kwargs))
        start_time = time.perf_counter()
        qtype, num_new_dots, classify_output = self.classify(classify_kwargs)
        print(f"Classify: {time.perf_counter() - start_time} seconds")
        print(classify_output)
        #if len(self.states) == 3:
        #    import pdb; pdb.set_trace()

        num_prev_dots = 0
        prev_dots = None
        all_dots = None
        previous_dots = None
        last_turn = None
        if qtype == Qtypes.FNEW or qtype == Qtypes.FOLD or qtype == Qtypes.SELECT:
            previous_dots, last_turn = self.get_last_confirmed_all_dots(self.states)
            num_prev_dots = previous_dots[0].sum().item()
            prev_dots = ",".join(letters[:num_prev_dots])
        dots = ",".join(letters[num_prev_dots:num_prev_dots+num_new_dots]) + ","
        if prev_dots is not None:
            prev_dots += ","
            all_dots = ",".join(letters[:num_prev_dots+num_new_dots]) + ","

        understand_blocks = constraints_dots()
        understand_kwargs = dict(
            header = HEADER,
            blocks = understand_blocks,
            speaker = speaker,
            text = text,
            type = qtype.value,
            prev_dots = prev_dots,
            dots = dots,
        )
        understand_prompt = self.understand.print(understand_kwargs)
        print(understand_prompt)

        import time
        start_time = time.perf_counter()
        constraints, savedots = self.understand(understand_kwargs)
        print(f"Understand: {time.perf_counter() - start_time} seconds")

        codeblock_dict = None
        if constraints is None:
            codeblock_dict = dict(
                noop = True,
                speaker = speaker,
                text = text,
                state = None,
            )
        elif qtype == Qtypes.NOOP:
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
        elif qtype == Qtypes.START:
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
                state = "None",
            )
        elif qtype == Qtypes.FOLD:
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
                state = [tuple(dots.nonzero()[0]) for dots in previous_dots],
            )
        elif qtype == Qtypes.FNEW:
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
                savedots = all_dots,
                state = [tuple(dots.nonzero()[0]) for dots in previous_dots],
            )
        elif qtype == Qtypes.SELECT:
            # construct codeblocks
            codeblock_dict = dict(
                noop = False,
                constraints = constraints,
                configs = "state",
                dots = prev_dots,
                newconfigs = "[0]",
                newdots = "_",
                select = "True",
                speaker = speaker,
                text = text,
                savedots = savedots,
                state = [tuple(dots.nonzero()[0]) for dots in previous_dots],
            )
        else:
            raise ValueError

        # new input for python execution
        kw = dict(
            info=info,
            header=HEADER,
            blocks=past.execute_past + [codeblock_dict],
            dots=view.tolist(),
            state = codeblock_dict["state"],
        )
        # TODO: CLEAN THIS UP. only execute code for THIS TURN.
        # results of previous turns are memoized.

        # debugging execution input
        input = self.execute.print(kw)
        print(input)

        result = self.execute(kw)
        print(result)
        print(f"Read after code: {time.perf_counter() - read_start_time} seconds")

        mentions = None
        if result is not None:
            num_preds = len(result)
            mentions = np.zeros((num_preds, 7), dtype=bool)
            for i in range(num_preds):
                mentions[i, result[i]] = 1

        plan = self.construct_plan(mentions, qtype, last_turn)
        return (
            plan,
            Past(
                classify_past = [],
                understand_past = [],
                execute_past = [],
            ),
            None,
        )

    def construct_plan(self, preds, qtype, refturn=None):
        # construct plan for what they said
        feats = None
        plan_idxs = None
        config_idx = None
        confirmed = None
        plan = None
        if preds is not None:
            # None => they didnt mention anything
            confirmed = preds.sum() > 0
            # sum == 0 => we dont see their plan
            if confirmed:
                feats = self.belief.get_feats(preds[0]) # pull out into fn
                plan_idxs = self.belief.resolve_utt(*feats) # pull out into fn
                config_idx = get_config_idx(preds[0], self.belief.configs)

            plan_dict = dict(
                dots = preds[0],
                config_idx = config_idx,
                feats = feats,
                plan_idxs = plan_idxs,
                all_dots = preds,
                confirmation = None, # FILL IN READ
                confirmed = confirmed,
                info_gain = None,
                qtype = qtype,
            )
            if qtype == Qtypes.START:
                plan_dict["new_dots"] = preds[0].sum().item()
                plan = StartPlan(**plan_dict)
            elif qtype == Qtypes.FOLD:
                last_state = self.states[refturn+1]
                plan_dict["newdots"] = preds[0] & last_state.plan.dots 
                plan_dict["olddots"] = preds[0] & ~last_state.plan.dots 
                plan_dict["new_dots"] = 0
                plan_dict["reference_turn"] = refturn
                plan = FollowupPlan(**plan_dict)
            elif qtype == Qtypes.FNEW:
                last_state = self.states[refturn+1]
                plan_dict["newdots"] = preds[0] & last_state.plan.dots 
                plan_dict["olddots"] = preds[0] & ~last_state.plan.dots 
                plan_dict["new_dots"] = plan_dict["newdots"].sum().item()
                plan_dict["reference_turn"] = refturn
                plan = FollowupPlan(**plan_dict)
            elif qtype == Qtypes.SELECT:
                plan_dict["new_dots"] = 0
                plan_dict["reference_turn"] = refturn
                plan = SelectPlan(**plan_dict)
            else:
                raise ValueError

        return plan
