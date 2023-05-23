import os
import time
import minichain
import numpy as np

import google.generativeai as palm

from oc.prompt import HEADER
from oc.agent.agent import Agent
from oc.dynamic_prompting.blocks import BLOCKS

palm.configure(api_key=os.getenv("GOOGLE_KEY"))

ctx = np.array([[-0.565, 0.775, 0.6666666666666666, -0.13333333333333333], [0.075, -0.715, 1.0, 0.16], [0.165, -0.58, 0.6666666666666666, -0.09333333333333334], [0.84, 0.525, 0.6666666666666666, -0.24], [0.655, -0.735, -0.6666666666666666, 0.44], [-0.31, -0.535, 0.6666666666666666, -0.48], [-0.03, -0.09, -0.6666666666666666, 0.9333333333333333]])

with minichain.start_chain("test-tmp.txt") as backend:
    agent = Agent(backend, "shortcodegen", "templateonly", "gpt-3.5-turbo")
    agent.feed_context(ctx.flatten().tolist())

    utterance = "Do you see a pair of dots, where the bottom left dot is large-sized and grey and the top right dot is large-sized and grey?"

    speaker = "Them"
    text = agent.reformat_text(utterance, usespeaker=False)
    past = []

    kwargs = dict(
        header=HEADER,
        blocks = BLOCKS,
        speaker = speaker,
        text=text,
        past=past,
        view=ctx,
    )

    understand_prompt = agent.understand.print(kwargs)
    print(understand_prompt)

    # GPT4
    start_time = time.perf_counter()
    codeblock = agent.understand(kwargs)
    print(f"GPT TIME: {time.perf_counter() - start_time} seconds")

    # GOOGLE
    start_time = time.perf_counter()
    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=understand_prompt,
        # The maximum length of the response
        max_output_tokens=1024,
        stop_sequences = ["# End."],
    )
    print(f"GOOGLE TIME: {time.perf_counter() - start_time} seconds")
    import pdb; pdb.set_trace()
