from abc import ABC, abstractmethod
import numpy as np
import evaluate


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
                import pdb

                pdb.set_trace()
                pred = self.predict(x)
                self.metric.add(prediction=pred, reference=label)
        return self.metric.compute()

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_labels(self, x):
        pass


def collapse_referents(xs):
    ret = np.ones(7)
    for x in xs:
        ret *= np.array(x["target"])
    return ret


class Resolution(Eval):
    metric = evaluate.load("accuracy")

    def predict(self, x):
        import pdb

        pdb.set_trace()

    def get_labels(self, example):
        referents = example["all_referents"]
        # collapse the referents in each turn
        return [collapse_referents(xs) for xs in referents]
        return


class Generation(Eval):
    metric = evaluate.load("bleu")

    def predict(self, x):
        import pdb

        pdb.set_trace()

    def get_labels(self, example):
        return example["dialogue"]


if __name__ == "__main__":
    from ocdata import get_data
    from agent import Agent, State

    data = get_data()
    agent = Agent()
    reseval = Resolution().compute(agent, data)
    import pdb

    pdb.set_trace()
