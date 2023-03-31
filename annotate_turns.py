import json
from pathlib import Path
import streamlit as st
import numpy as np

from ocdata import get_data
from dot import Dot, visualize_board, visualize_single_board

st.set_page_config(layout="wide")

# visualize logging information
with open('data/scenarios.json', "r") as f:
    scenario_list = json.load(f)
boards = {
    scenario['uuid']: scenario
    for scenario in scenario_list
}

_, data = get_data()

logdir = Path(f"resolution_logs/0/parsecodegen")
logfiles = list(sorted(logdir.iterdir()))
#print(logfiles)

st.write(f"Num examples: {len(data)}")

example_idx = st.number_input("example", 0, len(data)-1)

example = data[example_idx]
chat_id = example["chat_id"]

matching_logfiles = [f for f in logfiles if chat_id in str(f)]
logfile = matching_logfiles[0] if len(matching_logfiles) > 0 else None

if logfile is not None:
    with logfile.open("r") as f:
        log = json.load(f)

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

turns = example["dialogue"]
preds = None
raw_labels = [[x["target"] for x in xs] for xs in example["all_referents"]]
# the last turn might differ from log_labels because of selection
labels = [np.any(x, 0) for x in raw_labels]
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
        st.write(past[s][0])
        st.code(past[s][1])
        st.code("\n".join([
            f"{i}. " + ", ".join([dot_ids[i] for i,x in enumerate(pred) if x])
            for i, pred in enumerate(preds[t])
        ]))

col1, col2 = st.columns(2)
with col1:
    visualize_single_board(board, showlabel=True)

with col2:
    st.write("### Turn")
    st.code(turns[t])

    if logfile is not None:
        st.write("### ProcTurn")
        st.code(log_past[t][0])

        st.write("### Code")
        st.code(log_past[t][1])

    st.write("### Label")
    st.code(" ".join([dot_ids[i] for i,x in enumerate(labels[t]) if x]))
