import json
from pathlib import Path
import bitutils

from eval import Recall

recall = Recall("multilabel")

preds = []
labels = []

logdir = Path("resolution_logs")
logfiles = list(sorted(logdir.iterdir()))
for path in logfiles:
    with path.open("r") as f:
        log = json.load(f)
    lpreds = [[int(bitutils.config_to_int(pred)) for pred in preds] for preds in log["preds"]]
    llabels = [[int(bitutils.config_to_int(x))] for x in log["labels"]]

    preds.extend(lpreds)
    labels.extend(llabels)

print(recall.compute(predictions=preds, references=labels, average="micro"))
import pdb; pdb.set_trace()
