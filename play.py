from oc.dot import Dot, visualize_board, visualize_single_board

if __name__ == "__main__":
    from pathlib import Path
    import json
    import streamlit as st
    from streamlit_chat import message

    st.set_page_config(layout="wide")

    # Initialize chat state
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # visualize logging information
    with open('oc/data/scenarios.json', "r") as f:
        scenario_list = json.load(f)
    boards = {
        scenario['uuid']: scenario
        for scenario in scenario_list
    }

    scenario_id = scenario_list[0]["uuid"]

    board = boards[scenario_id]

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
    intersect1 = [x for x in b1 for y in b0 if x.id == y.id]
    mentions0 = None
    mentions1 = None

    # human will always be agent 0
    agent = 0

    board = b0 if agent == 0 else b1


    if 'input' not in st.session_state:
        st.session_state.input= ''

    def submit():
        st.session_state.input = st.session_state.widget
        st.session_state.widget = ''

    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    #with st.sidebar:
    st.write(f"### Scenario {scenario_id}")
    visualize_board(b0, b1, mentions0, mentions1, intersect0, intersect1)
    visualize_single_board(board, showlabel=False)

    placeholder = st.empty()
    user_input = st.text_input("You: ", key="widget", on_change=submit)

    if st.session_state.input:
        output = "lol"

        st.session_state.past.append(st.session_state.input)
        st.session_state.generated.append(output)

    with placeholder.container():
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))


