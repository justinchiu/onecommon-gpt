import datasets
from pathlib import Path
import numpy as np

EOS_TOKEN = "<eos>"
SELECTION_TOKEN = "<selection>"
YOU_TOKEN = "YOU:"
THEM_TOKEN = "THEM:"
SILENCE_TOKEN = "__SILENCE__"

INPUT_TAG = "input"
DIALOGUE_TAG = "dialogue"
REFERENTS_TAG = "referents"
PARTNER_REFERENTS_TAG = "partner_referents_our_view"
OUTPUT_TAG = "output"
REAL_IDS_TAG = "real_ids"

SCENARIO_TAG = "scenario_id"
CHAT_TAG = "chat_id"


REF_BEGIN_IDX = 0
REF_END_IDX = 1
REF_EOS_IDX = 2
REF_BEGIN_TARGET_IDX = 3

N_OBJECT = 7


def get_tag(tokens, tag):
    """
    Extracts the value inside the given tag.
    """
    start = tokens.index("<" + tag + ">") + 1
    stop = tokens.index("</" + tag + ">")
    return tokens[start:stop]


def _split_dialogue(words, separator=EOS_TOKEN):
    sentences = []
    spans = []
    start = 0
    for stop in range(len(words)):
        if words[stop] == separator:
            sentences.append(words[start:stop])
            spans.append((start, stop))
            start = stop + 1
    if stop >= start:
        sentences.append(words[start:])
        spans.append((start, len(words) - 1))

    # Dataset contains consecutive turn
    # concatenate utterances for those cases
    dialogue = []
    utterance = sentences[0]
    for i in range(1, len(sentences)):
        if sentences[i - 1][0] == sentences[i][0]:
            utterance += sentences[i][1:]
        else:
            dialogue.append(utterance)
            utterance = sentences[i]
    dialogue.append(utterance)

    """
    if dialogue[0][0] == YOU_TOKEN:
        # Dialogue starts with YOU
        dialogue.insert(0, None)
        spans.insert(0, None)
    if dialogue[-1][0] == THEM_TOKEN:
        # Dialogue starts with THEM
        dialogue.append(None)
        spans.append(None)
    """

    return dialogue, spans


def _split_referents(raw_referents, spans):
    """
    Split the referents.
    The first 3 values are begin idx, end idx, and eos idx The next N_OBJECT values
    are booleans of if the object is referred e.g. 3 4 10 0 1 0 0 0 0 0 means idx 3
    to 4 is a markable of an utterance with <eos> at idx 10, and it refers to the
    2nd dot
    """

    referent_len = 3 + N_OBJECT
    splitted_referents = []
    for i in range(len(raw_referents) // referent_len):
        val = raw_referents[i * referent_len : (i + 1) * referent_len]
        splitted_referents.append(list(map(int, val)))

    referents = []
    idx = 0
    for span in spans:
        if span is None:
            referents.append(None)
            continue

        # span is a (bos index, eos index) of an utterance
        refs = []
        while idx < len(splitted_referents):
            if splitted_referents[idx][REF_EOS_IDX] == span[1]:
                ref = {
                    "begin": splitted_referents[idx][REF_BEGIN_IDX] - (span[0] + 1),
                    "end": splitted_referents[idx][REF_END_IDX] - (span[0] + 1),
                    "target": splitted_referents[idx][REF_BEGIN_TARGET_IDX:],
                }
                refs.append(ref)
                idx += 1
            else:
                break
        referents.append(refs)

    return referents


def get_examples(raw_data):
    examples = []
    for data in raw_data:
        words = data.strip().split()
        # The linearized context values
        # Each dot is represented as (x, y, size, color)
        # There should be 28 values = 4 values * 7 dots.
        context = list(map(float, get_tag(words, INPUT_TAG)))
        dialogue, spans = _split_dialogue(get_tag(words, DIALOGUE_TAG))

        # all referents are in your view
        referents = _split_referents(get_tag(words, REFERENTS_TAG), spans)
        partner_referents = _split_referents(
            get_tag(words, PARTNER_REFERENTS_TAG), spans
        )

        output = int(get_tag(words, OUTPUT_TAG)[0])
        context = np.array(context).reshape((7, 4))
        # y-axis is inverted
        context[:,1] = -context[:,1]
        real_ids = get_tag(words, "real_ids")
        partner_real_ids = get_tag(words, "partner_real_ids")
        agent = int(get_tag(words, "agent")[0])
        scenario_id = get_tag(words, SCENARIO_TAG)[0]
        chat_id = get_tag(words, CHAT_TAG)[0]

        examples.append(
            {
                "context": context,
                "dialogue": [
                    " ".join(turn).replace("YOU", "You").replace("THEM", "Them")
                    if turn
                    else turn
                    for turn in dialogue
                ],
                "all_referents": [
                    a if len(a) > 0 else b for a, b in zip(referents, partner_referents)
                ],
                "referents": referents,
                "partner_referents": partner_referents,
                "output": output,
                "scenario_id": scenario_id,
                "chat_id": chat_id,
                "agent": agent,
                "real_ids": real_ids,
                "partner_real_ids": partner_real_ids,
            }
        )
    return examples


def get_data(split=1, filter_agent=True):
    datas = []
    for data_file in [
        Path(f"data/onecommon/train_reference_{split}.txt"),
        Path(f"data/onecommon/valid_reference_{split}.txt"),
    ]:
        with data_file.open("r") as f:
            raw_data = f.readlines()
            datas.append(get_examples(raw_data))
    # WARNING: WILL REMOVE REPEATS FROM DATA
    if filter_agent:
        datas = [[x for x in data if x["agent"] == 0] for data in datas]
    return datas


if __name__ == "__main__":
    train_data, valid_data = get_data()

    # make sure no bleeding into other sets
    _, valid_data0 = get_data(0)
    ids0 = set(x["chat_id"] for x in valid_data0)
    _, valid_data1 = get_data(1)
    ids1 = set(x["chat_id"] for x in valid_data1)
    _, valid_data2 = get_data(2)
    ids2 = set(x["chat_id"] for x in valid_data2)
    import pdb; pdb.set_trace()
