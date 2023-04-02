import json
from pathlib import Path
import streamlit as st
import numpy as np
from collections import defaultdict

from ocdata import get_data
from dot import Dot, visualize_board, visualize_single_board

st.set_page_config(layout="wide")

annotation_dir = Path("annotation")
annotation_dir.mkdir(parents=True, exist_ok=True)

# visualize logging information
with open('data/scenarios.json', "r") as f:
    scenario_list = json.load(f)
boards = {
    scenario['uuid']: scenario
    for scenario in scenario_list
}

_, data = get_data()

logdir = Path(f"resolution_logs/0/parsecodegen")
logfiles = [x for x in sorted(logdir.iterdir()) if "agent0" in str(x)]
#print(logfiles)

st.write(f"Num examples: {len(data)}")

example_idx = st.number_input("example", 0, len(data)-1)
example_idx = 3

example = data[example_idx]
chat_id = example["chat_id"]


# annotation file
annotation_file = (annotation_dir / chat_id).with_suffix(".json") 

if annotation_file.exists():
    with annotation_file.open("r") as f:
        annotation = json.load(f)
else:
    annotation = defaultdict(str)
# / annotation files
 

# model log files 
matching_logfiles = [f for f in logfiles if chat_id in str(f)]
logfile = matching_logfiles[0] if len(matching_logfiles) > 0 else None

if logfile is not None:
    with logfile.open("r") as f:
        log = json.load(f)
# / model log files


st.write(f"### Dialogue {log['chat_id']}")
st.write(f"### Scenario {log['scenario_id']}")

board = boards[log["scenario_id"]]

b0 = [Dot(x) for x in board["kbs"][0]]
b1 = [Dot(x) for x in board["kbs"][1]]
intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
intersect1 = [x for x in b1 for y in b0 if x.id == y.id]
mentions0 = None
mentions1 = None

if logfile is not None:
    log_turns = log["turns"]
    log_preds = log["preds"]
    log_labels = log["labels"]
    log_past = log["past"]
    log_agent = log["agent"]
    log_dot_ids = log["dot_ids"]

    log_parsedtext = log["parsedtext"] if "parsedtext" in log else None
    log_bullet = log["bullet"] if "bullet" in log else None
    log_output = log["output"] if "output" in log else None

turns = example["dialogue"]
preds = None
raw_labels = [[x["target"] for x in xs] for xs in example["all_referents"]]

# the last turn might differ from log_labels because of selection
labels = [
    np.any(x, 0) if len(x) > 0 else np.zeros(7, dtype=bool)
    for x in raw_labels
]
# last turn is selection
labels[-1][example["output"]] = True

agent = example["agent"]
dot_ids = example["real_ids"]

print(dot_ids)
print(agent)

board = b0 if agent == 0 else b1

st.write(f"Num turns: {len(turns)}")
t = st.number_input("turn", 0, len(turns)-1)

#visualize_board(b0, b1, mentions0, mentions1, intersect0, intersect1)

with st.sidebar:
    st.write("# Past")
    for s in range(t):
        st.write(f"### Turn {s}")
        if log_past is not None:
            st.write(log_past[s][0])
            st.code(log_past[s][1])
            st.code("\n".join([
                f"{i}. " + ", ".join([dot_ids[i] for i,x in enumerate(pred) if x])
                for i, pred in enumerate(log_preds[t])
            ]))

col1, col2 = st.columns(2)
with col1:
    visualize_single_board(board, showlabel=True)
    st.write("### Label")
    st.code(" ".join([dot_ids[i] for i,x in enumerate(labels[t]) if x]))
    st.write("### Turn")
    st.code(turns[t])

with col2:
    if logfile is not None:
        if log_parsedtext is not None:
            st.write("### ProcTurn")
            st.code(log_parsedtext[t])

        if log_bullet is not None:
            st.write("### Bullet")
            st.code(log_bullet[t])

        st.write("### Code")
        st.code(log_past[t][1])

with st.form("annot"):
    bullet = st.text_area("Bullet", value=annotation["bullet"])
    code = st.text_area("Code", value=annotation["code"])

    annotation = {
        "bullet": bullet,
        "code": code,
    }

    submitted = st.form_submit_button("Submit")
    if submitted:
        with annotation_file.open("w") as f:
            json.dump(annotation, f)
