from abc import ABC, abstractmethod
import numpy as np
import evaluate
import bitutils
import pdb


class Eval(ABC):
    def compute(self, agent, data, num_examples=None):
        for example in data[:num_examples]:
            chatid = example["chat_id"]
            scenarioid = example["scenario_id"]
            print(scenarioid)
            print(chatid)

            view = example["context"]
            turns = example["dialogue"]
            labels = self.get_labels(example)
            past = []
            for t in range(len(turns)):
                text = turns[t]
                label = labels[t]
                pred, past = self.predict(agent, text, past, view)
                #print(pred, label)
                #import pdb; pdb.set_trace()
                self.metric.add(prediction=pred, reference=label)
        return self.metric.compute()

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_labels(self, x):
        pass


def collapse_referents(xs):
    ret = np.zeros(7, dtype=bool)
    for x in xs:
        ret |= np.array(x["target"], dtype=bool)
    return bitutils.config_to_int(ret)


class Resolution(Eval):
    metric = evaluate.load("accuracy")

    def predict(self, agent, text, past, view):
        pred, newpast = agent.resolve_reference(text, past, view)
        return bitutils.config_to_int(pred), newpast

    def get_labels(self, example):
        referents = example["all_referents"]
        # collapse the referents in each turn
        return [collapse_referents(xs) for xs in referents]


class Generation(Eval):
    metric = evaluate.load("bleu")

    def predict(self, agent, text, past, view):
        plan = agent.plan(past, view)
        return agent.generate(plan, past, view)

    def get_labels(self, example):
        return example["dialogue"]


if __name__ == "__main__":
    from ocdata import get_data
    from agent import Agent, State
    import minichain

    data = get_data()
    with minichain.start_chain("eval-res") as backend:
        agent = Agent(backend)
        reseval = Resolution().compute(agent, data, 1)
    print(reseval)

    with minichain.start_chain("eval-gen") as backend:
        agent = Agent(backend)
        geneval = Generation().compute(agent, data, 1)
    print(geneval)


    pdb.set_trace()
