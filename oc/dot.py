class Dot:
    def __init__(self, item):
        for k, v in item.items():
            setattr(self, k, v)
        self.id = int(self.id)

    def html(self, shift=0, value=None, showlabel=True):
        x = self.x + shift
        y = self.y
        r = self.size
        f = self.color
        if showlabel:
            label = (
                f'<text x="{x+12}" y="{y-12}" font-size="18">{self.id}</text>'
                if value is None
                else f'<text x="{x+12}" y="{y-12}" font-size="18">{self.id} ({value:.2f})</text>'
            )
            return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" /> {label}'
        else:
            return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" />'

    def select_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 8
        f = self.color  # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="red" stroke-width="3" stroke-dasharray="3,3"  />'

    def intersect_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 4
        f = self.color  # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="blue" stroke-width="3" stroke-dasharray="3,3"  />'

    def __repr__(self):
        return f"Dot {self.id}: ({self.x}, {self.y}) r={self.size} f={self.color}"


def visualize_board(
    left_dots,
    right_dots,
    left_mentions,
    right_mentions,
    left_intersect,
    right_intersect,
    left_beliefs=None,
    right_beliefs=None,
):
    import streamlit as st
    import streamlit.components.v1 as components

    shift = 430

    left_dots_html = (
        map(lambda x: x.html(), left_dots)
        if left_beliefs is None
        else map(lambda x: x[0].html(value=x[1]), zip(left_dots, left_beliefs))
    )
    right_dots_html = (
        map(lambda x: x.html(shift), right_dots)
        if right_beliefs is None
        else map(lambda x: x[0].html(shift, value=x[1]), zip(right_dots, right_beliefs))
    )

    if left_mentions is not None:
        left_mentions_html = map(lambda x: x.select_html(), left_mentions)
    if right_mentions is not None:
        right_mentions_html = map(lambda x: x.select_html(shift), right_mentions)
    if left_intersect is not None:
        left_intersect_dots = map(lambda x: x.intersect_html(), left_intersect)
    if right_intersect is not None:
        right_intersect_dots = map(lambda x: x.intersect_html(shift), right_intersect)

    nl = "\n"
    html = f"""
    <svg width="860" height="430">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(left_dots_html)}
    {nl.join(left_intersect_dots) if left_intersect is not None else ""}
    {nl.join(left_mentions_html) if left_mentions is not None else ""}
    <circle cx="645" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(right_dots_html)}
    {nl.join(right_intersect_dots) if right_intersect is not None else ""}
    {nl.join(right_mentions_html) if right_mentions is not None else ""}
    </svg>
    """
    components.html(html, height=430, width=860)

def visualize_single_board(dots, showlabel=False):
    import streamlit as st
    import streamlit.components.v1 as components
    dots_html = map(lambda x: x.html(showlabel = showlabel), dots)

    nl = "\n"
    html = f"""
    <svg width="450" height="500" overflow="visible">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(dots_html)}
    </svg>
    """
    components.html(html, height=500, width=450)


if __name__ == "__main__":
    from pathlib import Path
    import json
    import streamlit as st
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--method",
        choices = ["codegen", "parsecodegen"],
        default = "codegen",
    )
    parser.add_argument("--model",
        choices = ["gpt-4", "gpt-3.5-turbo"],
        default = "gpt-4",
    )
    parser.add_argument("--data",
        choices = ["valid", "train"],
        default = "valid",
    )
    args = parser.parse_args()

    st.set_page_config(layout="wide")

    # visualize logging information
    with open('data/scenarios.json', "r") as f:
        scenario_list = json.load(f)
    boards = {
        scenario['uuid']: scenario
        for scenario in scenario_list
    }

    logdir = Path(f"resolution_logs/{args.split}/{args.data}/{args.model}/{args.method}")
    logfiles = list(sorted(logdir.iterdir()))
    #print(logfiles)

    st.write(f"Num examples: {len(logfiles)}")
    log_idx = st.number_input("example", 0, len(logfiles)-1)
    logfile = logfiles[log_idx]

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

    turns = log["turns"]
    preds = log["preds"]
    labels = log["labels"]
    past = log["past"]
    agent = log["agent"]
    dot_ids = log["dot_ids"]
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

        st.write("### ProcTurn")
        st.code(past[t][0])

        st.write("### Code")
        st.code(past[t][1])

        st.write("### Label")
        st.code(" ".join([dot_ids[i] for i,x in enumerate(labels[t]) if x]))

    with col1:
        st.write("### Preds")
        print(preds[t])
        st.code("\n".join([
            f"{i}. " + ", ".join([dot_ids[i] for i,x in enumerate(pred) if x])
            for i, pred in enumerate(preds[t])
        ]))

