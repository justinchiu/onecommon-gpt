import pytest
from unittest.mock import patch
from oc.agent.agent import Agent

class TestAgent:
    def test_run(self, mocked_input):
        dialogue_system = DialogueSystem()
        dialogue_system.write = lambda message: message # Override write method for testing
        assert dialogue_system.run() == 'Hello!'

    def test_plan(self):
        dialogue_system = DialogueSystem()
        assert dialogue_system.plan('Hello!') == 'Hello!'

    def test_read(self, monkeypatch):
        monkeypatch.setattr('builtins.input', lambda _: 'Hello!')
        dialogue_system = DialogueSystem()
        assert dialogue_system.read() == 'Hello!'

