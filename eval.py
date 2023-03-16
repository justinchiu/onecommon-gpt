from abc import ABC, abstractmethod
import numpy as np
import evaluate
import bitutils
import pdb


class Eval(ABC):
    def compute(self, agent, data, num_examples=None):
        preds = []
        truelabels = []
        for example in data[:num_examples]:
            chatid = example["chat_id"]
            scenarioid = example["scenario_id"]
            print(scenarioid)
            print(chatid)

            view = example["context"]
            turns = example["dialogue"]
            referents = example["all_referents"]
            labels = self.get_labels(example)
            past = []
            for t in range(len(turns)):
                text = turns[t]
                past_turns = turns[:t]
                plan = referents[t]

                input = dict(
                    agent = agent,
                    text = text,
                    past = past,
                    view = view,
                    past_turns = past_turns,
                    plan = plan,
                    info = (scenarioid, chatid),
                )

                pred, past = self.predict(**input)

                label = labels[t]
                #import pdb; pdb.set_trace()
                if self.do_eval(text):
                    preds.append(pred)
                    truelabels.append(label)
                    print("LABEL")
                    print(label)

        return self.metric.compute(predictions=preds, references=truelabels)

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_labels(self, x):
        pass

    @abstractmethod
    def do_eval(self, x):
        pass


def collapse_referents(xs):
    ret = np.zeros(7, dtype=bool)
    for x in xs:
        ret |= np.array(x["target"], dtype=bool)
    return bitutils.config_to_int(ret)


class Resolution(Eval):
    metric = evaluate.load("accuracy")

    def predict(self, agent, text, past, view, plan, past_turns, info=None):
        pred, newpast = agent.resolve_reference(text, past, view, info)
        return bitutils.config_to_int(pred), newpast

    def get_labels(self, example):
        referents = example["all_referents"]
        # collapse the referents in each turn
        return [collapse_referents(xs) for xs in referents]

    def do_eval(self, turn):
        return True


class Generation(Eval):
    metric = evaluate.load("bleu")

    def predict(self, agent, text, past, view, plan, past_turns, info=None):
        #plan = agent.plan(past, view, info)
        return agent.generate_text(plan, past_turns, view, info)

    def get_labels(self, example):
        return example["dialogue"]

    def do_eval(self, turn):
        return turn.split()[0] == "You:"


if __name__ == "__main__":
    from ocdata import get_data
    from agent import Agent, State
    import minichain

    train, valid = get_data()
    with minichain.start_chain("eval-res") as backend:
        agent = Agent(backend)
        reseval = Resolution().compute(agent, valid, 1)
    print(reseval)

    with minichain.start_chain("eval-gen") as backend:
        agent = Agent(backend)
        geneval = Generation().compute(agent, valid, 5)
    print(geneval)


    pdb.set_trace()
