import json
from pathlib import Path
import bitutils

from eval import Recall

#split = "train"
split = "valid"
#model = "gpt-3.5-turbo"
model = "gpt-4"
method = "codegen"
#method = "parsecodegen"

print(f"Evaluating split {split}")

recall = Recall("multilabel")

preds = []
labels = []

logdir = Path(f"resolution_logs/1/{split}/{model}/{method}")
logfiles = list(sorted(logdir.iterdir()))
for path in logfiles:
    with path.open("r") as f:
        log = json.load(f)
    lpreds = log["preds"]
    llabels = [[x] for x in log["labels"]]
    print(path)
    print(recall.compute(predictions=lpreds, references=llabels))
    preds.extend(lpreds)
    labels.extend(llabels)

print("global")
metrics = recall.compute(predictions=preds, references=labels)
for k,v in metrics.items():
    print(f"{k}: {v}")
