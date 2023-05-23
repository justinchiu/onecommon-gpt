import pytest
from unittest.mock import patch

import numpy as np
import minichain

from oc.agent.agent import Agent
from oc.agent2.agent import Agent as Agent2

class TestAgent:
    def test_read_shortcodegen2(self):
        ctx = np.array([[-0.565, 0.775, 0.6666666666666666, -0.13333333333333333], [0.075, -0.715, 1.0, 0.16], [0.165, -0.58, 0.6666666666666666, -0.09333333333333334], [0.84, 0.525, 0.6666666666666666, -0.24], [0.655, -0.735, -0.6666666666666666, 0.44], [-0.31, -0.535, 0.6666666666666666, -0.48], [-0.03, -0.09, -0.6666666666666666, 0.9333333333333333]])

        with minichain.start_chain("test-tmp.txt") as backend:
            agent1 = Agent(backend, "shortcodegen", "templateonly", "gpt-4")
            agent1.feed_context(ctx.flatten().tolist())
            agent2 = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")
            agent2.feed_context(ctx.flatten().tolist())

            utterance = "Do you see a pair of dots, where the bottom left dot is large-sized and grey and the top right dot is large-sized and grey?"
            agent1.read(utterance)
            agent2.read(utterance)

            dots1 = agent1.preds[-1][0]
            dots2 = agent2.states[-1].plan.dots
            assert (dots1 == dots2).all()

            utt1 = agent1.write()
            utt2 = agent2.write()
            assert utt1 == utt2

            dots21 = agent1.plans[-1].dots
            dots22 = agent2.states[-1].plan.dots
            assert (dots21 == dots22).all()

            utterance = "No. Is there a large-size grey dot to the left of those though?"
            agent1.read(utterance)
            agent2.read(utterance)
            dots31 = agent1.preds[-1][0]
            dots32 = agent2.states[-1].plan.dots
            assert (dots31 == dots32).all()
            # might actually be different, due to parsing of "pair"

if __name__ == "__main__":
    TestAgent().test_read_shortcodegen2()
