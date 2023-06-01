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

            utterance = "Do you see a pair of dots, where the bottom left dot is large-sized and grey and the top right dot is large-sized and grey?".split()
            agent2.read(utterance)
            agent1.read(utterance)

            dots1 = agent1.preds[-1][0]
            dots2 = agent2.states[-1].plan.dots
            print(dots1)
            print(dots2)
            assert (dots1 == dots2).all()

            utt1 = agent1.write()
            utt2 = agent2.write()
            print(utt1)
            print(utt2)
            assert utt1 == utt2

            dots21 = agent1.plans[-1].dots
            dots22 = agent2.states[-1].plan.dots
            print(dots21)
            print(dots22)
            assert (dots21 == dots22).all()

            utterance = "No. Is there a large-size grey dot to the left of those though?".split()
            agent1.read(utterance)
            agent2.read(utterance)
            print(agent1.preds[-1])
            #dots31 = agent1.preds[-1][0]
            dots32 = agent2.states[-1].plan.dots
            #print(dots31)
            print(dots32)
            #import pdb; pdb.set_trace()
            #assert (dots31 == dots32).all()
            # might actually be different, due to parsing of "pair"
            print(agent2.write())

    def test_consistent_shortcodegen2(self):
        ctx = np.array([[-0.565, 0.775, 0.6666666666666666, -0.13333333333333333], [0.075, -0.715, 1.0, 0.16], [0.165, -0.58, 0.6666666666666666, -0.09333333333333334], [0.84, 0.525, 0.6666666666666666, -0.24], [0.655, -0.735, -0.6666666666666666, 0.44], [-0.31, -0.535, 0.6666666666666666, -0.48], [-0.03, -0.09, -0.6666666666666666, 0.9333333333333333]])

        with minichain.start_chain("test-tmp.txt") as backend:
            agent = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")
            agent.feed_context(ctx.flatten().tolist())

            utterance = "Do you see a pair of dots, where the bottom left dot is large-sized and grey and the top right dot is large-sized and grey?".split()
            agent.read(utterance)

            dots2 = agent.states[-1].plan.dots
            print(dots2)

            utt2 = agent.write()
            print(utt2)

            dots22 = agent.states[-1].plan.dots
            print(dots22)

            utterance = "No. Is there a large-size grey dot to the left of those though?".split()
            agent.read(utterance)

            dots32 = agent.states[-1].plan.dots
            print(dots32)

            print(agent.write())

    def test_select_shortcodegen2(self):
        ctx = np.array([[-0.565, 0.775, 0.6666666666666666, -0.13333333333333333], [0.075, -0.715, 1.0, 0.16], [0.165, -0.58, 0.6666666666666666, -0.09333333333333334], [0.84, 0.525, 0.6666666666666666, -0.24], [0.655, -0.735, -0.6666666666666666, 0.44], [-0.31, -0.535, 0.6666666666666666, -0.48], [-0.03, -0.09, -0.6666666666666666, 0.9333333333333333]])

        with minichain.start_chain("test-tmp.txt") as backend:
            agent = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")
            agent.feed_context(ctx.flatten().tolist())

            utt = agent.write()
            utterance = "Yes".split()
            agent.read(utterance)
            utt = agent.write()
            utterance = "Yes".split()
            agent.read(utterance)
            utt = agent.write()
            utterance = "Yes".split()
            agent.read(utterance)

    def test_follow_shortcodegen2(self):
        ctx = np.array([[-0.565, 0.775, 0.6666666666666666, -0.13333333333333333], [0.075, -0.715, 1.0, 0.16], [0.165, -0.58, 0.6666666666666666, -0.09333333333333334], [0.84, 0.525, 0.6666666666666666, -0.24], [0.655, -0.735, -0.6666666666666666, 0.44], [-0.31, -0.535, 0.6666666666666666, -0.48], [-0.03, -0.09, -0.6666666666666666, 0.9333333333333333]])

        with minichain.start_chain("test-tmp.txt") as backend:
            agent = Agent2(backend, "shortcodegen2", "templateonly", "gpt-4")
            agent.feed_context(ctx.flatten().tolist())

            utterance = "Do you see a pair of dots, where the bottom left dot is large-sized and grey and the top right dot is large-sized and grey?".split()
            agent.read(utterance)

            dots1 = agent.states[-1].plan.dots
            print(dots1)

            utt1 = agent.write()
            print(utt1)

            dots2 = agent.states[-1].plan.dots
            print(dots2)

            utterance = "No. Is there a large-size grey dot to the left of those though?".split()
            utterance = "No. Is there a small-size grey dot to the left of those though?".split()
            agent.read(utterance)
            dots3 = agent.states[-1].plan
            print(dots3)
            start_plan = agent.plan_start(agent.states)
            follow_plan = agent.plan_followup(agent.states)

            start = agent.generate_new_config(start_plan, None, ctx)
            follow = agent.generate_followup(follow_plan, None, ctx)

            print("start")
            print(start_plan.dots)
            print(start_plan.info_gain)
            print(start)
            print("follow")
            print(follow_plan.dots)
            print(follow_plan.info_gain)
            print(follow)
            print(agent.belief.marginals(agent.states[-1].belief_dist))

            # try manually asking about 1 new dot
            belief_dist = agent.states[-1].belief_dist
            last_confirmed_dots, confirmed_turn = agent.get_last_confirmed_dots(agent.states)
            config_mask = ((agent.belief.configs & last_confirmed_dots) == last_confirmed_dots).all(-1)
            repeat_mask = agent.get_repeat_mask(agent.states)
            one_new_mask = agent.belief.configs.sum(-1) == (last_confirmed_dots.sum() + 1)

            EdHs = agent.belief.compute_EdHs(belief_dist)
            (EdHs * config_mask * repeat_mask * one_new_mask)

            import pdb; pdb.set_trace()
            # might actually be different, due to parsing of "pair"
            print(agent.write())

if __name__ == "__main__":
    #TestAgent().test_read_shortcodegen2()
    #TestAgent().test_consistent_shortcodegen2()
    #TestAgent().test_select_shortcodegen2()
    TestAgent().test_follow_shortcodegen2()
