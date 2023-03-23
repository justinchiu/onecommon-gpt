from dot import Dot, visualize_board, visualize_single_board

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text
         
if __name__ == "__main__":
    from pathlib import Path
    import json
    import streamlit as st
    from streamlit_chat import message

    # Initialize chat state
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # visualize logging information
    with open('data/scenarios.json', "r") as f:
        scenario_list = json.load(f)
    boards = {
        scenario['uuid']: scenario
        for scenario in scenario_list
    }

    scenario_id = scenario_list[0]["uuid"]

    st.write(f"### Scenario {scenario_id}")

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

    #visualize_board(b0, b1, mentions0, mentions1, intersect0, intersect1)
    visualize_single_board(board, showlabel=False)

    placeholder = st.empty()
    user_input = get_text()
    if user_input:
        output = {"generated_text": "lol"}

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["generated_text"])

    with placeholder.container():
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
