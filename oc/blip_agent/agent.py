import json
from PIL import Image
from pathlib import Path
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import imgkit

from oc.prompt import Reformat
from oc.dot import Dot, single_board_html

from importlib.resources import files
import oc.data
DATA_DIR = Path(files(oc.data)._paths[0])


class BlipAgent():
    def __init__(self, backend):
        with (DATA_DIR / "scenarios.json").open("r") as f:
            self.scenarios = json.load(f)
            self.boards = {
                scenario['uuid']: scenario
                for scenario in self.scenarios
            }

        self.reformat = Reformat(backend.OpenAIChat(
            model = "gpt-3.5-turbo-0613",
            max_tokens = 128,
        ))
        self.turns = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        self.model.to(device)

    def feed_context(self, ctx, flip_y, belief_constructor=None, scenario_id=None):
        dots = [Dot(x) for x in board["kbs"][0]]
        html = single_board_html(dots)
        import pdb; pdb.set_trace()
        self.ctx = Image()

    def read(self, input_words):
        raw_text = " ".join(input_words)
        text = self.reformat(dict(source=raw_text)).strip()
        self.turns.append(text)

        image = self.ctx

        prompt = "Question: how many cats are there? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        import pdb; pdb.set_trace()

    def write(self):
        pass
