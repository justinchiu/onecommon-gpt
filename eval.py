from abc import ABC, abstractmethod
import evaluate


class Eval(ABC):
    def evaluate(self, data):
        metric = self.metric()
        for example in data:
            for turn in turns:
                x = 1
                label = 1
                pred = self.predict(x)
                metric.add(prediction=pred, reference=label)
                import pdb; pdb.set_trace()
        return metric.compute()

    @abstractmethod
    def predict(self, x):
        pass

class Resolution(Eval):
    metric = evaluate.load("accuracy")
    def predict(self, x):
        import pdb; pdb.set_trace()

class Generation(Eval):
    metric = load_metric("bleu")
    def predict(self, x):
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    import pdb; pdb.set_trace()
