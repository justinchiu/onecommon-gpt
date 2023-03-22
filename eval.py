from abc import ABC, abstractmethod
import numpy as np
import datasets
import evaluate
import bitutils
import pdb

class Recall(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html"],
        )

    def _compute(
        self,
        predictions,
        references,
        labels=None,
        pos_label=1,
        average="sample",
        sample_weight=None,
        zero_division="warn",
    ):
        p = 0
        tp = 0
        num_preds = 0
        for preds, labels in zip(predictions, references):
            p += len(labels)
            num_preds += len(preds)
            for pred in preds:
                if pred in labels:
                    tp += 1
        return {"recall": tp / p, "precision": tp / num_preds}

class Eval(ABC):
    flags = dict()

    def compute(self, agent, data, num_examples=None):
        configs = bitutils.get_configs(128)
        preds = []
        truelabels = []
        for ne, example in enumerate(data[:num_examples]):
            chatid = example["chat_id"]
            scenarioid = example["scenario_id"]
            print(f"Example {ne}")
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
                    print("LABEL")
                    if isinstance(self, Generation):
                        preds.append(pred)
                        truelabels.append(label)
                        print(label)
                    elif isinstance(self, Resolution):
                        preds.append(pred)
                        truelabels.append([label])
                        print(configs[label].nonzero()[0])

        return self.metric.compute(predictions=preds, references=truelabels, **self.flags)

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
    return int(bitutils.config_to_int(ret))


class Resolution(Eval):
    #metric = evaluate.load("recall", "multilabel")
    metric = Recall("multilabel")
    flags = dict(average="micro")

    def predict(self, agent, text, past, view, plan, past_turns, info=None):
        pred, newpast = agent.resolve_reference(text, past, view, info)
        intpreds = bitutils.config_to_int(pred)
        out = list(set(intpreds.tolist()))
        if len(out) == 0:
            # fill with no reference
            out = [0]
        return out, newpast

    def get_labels(self, example):
        referents = example["all_referents"]
        # collapse the referents in each turn
        referents = [collapse_referents(xs) for xs in referents]
        # final turn is selection. output instead of mentions
        referents[-1] = example["output"]
        return referents

    def do_eval(self, turn):
        #if "<selection>" in turn:
        #    return False
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

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refres", choices=["codegen", "mc"], default="mc")
    parser.add_argument("--gen",
        choices=["sc", "scxy", "template", "templateonly"],
        default="template",
    )
    parser.add_argument("--run_refres", action="store_true")
    parser.add_argument("--run_gen", action="store_true")
    parser.add_argument("--num_examples", default=1, type=int)
    args = parser.parse_args()

    refres = args.refres
    gen = args.gen

    train, valid = get_data()

    if args.run_refres:
        with minichain.start_chain("eval-res") as backend:
            agent = Agent(backend, refres, gen)
            reseval = Resolution().compute(agent, valid, args.num_examples)
        print(reseval)

    if args.run_gen:
        with minichain.start_chain("eval-gen") as backend:
            agent = Agent(backend, refres, gen)
            geneval = Generation().compute(agent, valid, args.num_examples)
        print(geneval)

