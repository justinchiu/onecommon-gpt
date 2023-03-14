import openai

from prompt import Understand, Generate


@dataclass
class State:
    memory: List[Tuple[str, str]]
    human_input: str = ""

    def push(self, response: str) -> "State":
        memory = self.memory if len(self.memory) < MEMORY else self.memory[1:]
        return State(memory + [(self.human_input, response)])


class Agent:
    def __init__(self):
        self.understand = Understand()

    def read(self):
        pass

    def write(self):
        pass

    def resolve_reference(text, past, view):
        pass

    def plan(past, view):
        pass

    def generate_text(past, view):
        pass
