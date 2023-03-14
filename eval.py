from abc import ABC, abstractmethod
import numpy as np
import evaluate
import pdb


class Eval(ABC):
    def compute(self, agent, data):
        for example in data:
            view = example["context"]
            turns = example["dialogue"]
            labels = self.get_labels(example)
            for t in range(len(turns)):
                text = turns[t]
                past = turns[:t]
                label = labels[t]
                pred = self.predict(agent, text, past, view)
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
    return ret


class Resolution(Eval):
    metric = evaluate.load("accuracy")

    def predict(self, agent, text, past, view):
        return agent.resolve_reference(text, past, view)

    def get_labels(self, example):
        referents = example["all_referents"]
        # collapse the referents in each turn
        return [collapse_referents(xs) for xs in referents]


class Generation(Eval):
    metric = evaluate.load("bleu")

    def predict(self, agent, text, past, view):
        plan = agent.plan(past, view)
        return agent.generate(plan, past, view)
        pdb.set_trace()

    def get_labels(self, example):
        return example["dialogue"]


if __name__ == "__main__":
    from ocdata import get_data
    from agent import Agent, State

    data = get_data()
    agent = Agent()
    reseval = Resolution().compute(agent, data)
    pdb.set_trace()
