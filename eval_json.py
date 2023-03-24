import json
from pathlib import Path
import bitutils

from eval import Recall

split = 0

print(f"Evaluating split {split}")

recall = Recall("multilabel")

preds = []
labels = []

logdir = Path(f"resolution_logs/{split}")
logfiles = list(sorted(logdir.iterdir()))
for path in logfiles:
    with path.open("r") as f:
        log = json.load(f)
    lpreds = [[int(bitutils.config_to_int(pred)) for pred in preds] for preds in log["preds"]]
    llabels = [[int(bitutils.config_to_int(x))] for x in log["labels"]]

    print(path)
    print(recall.compute(predictions=lpreds, references=llabels, average="micro"))
    preds.extend(lpreds)
    labels.extend(llabels)

print("global")
print(recall.compute(predictions=preds, references=labels, average="micro"))
