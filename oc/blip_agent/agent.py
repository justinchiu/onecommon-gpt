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

    def feed_context(self, ctx, flip_y, belief_constructor=None, scenario_id=None, agent_id=None):
        assert scenario_id is not None
        assert agent_id is not None

        path = DATA_DIR / scenario_id / f"{agent_id}.jpg"
        self.image = Image.open(str(path))

    def read(self, input_words, resolve_references=False):
        raw_text = " ".join(input_words)
        text = self.reformat(dict(source=raw_text)).strip()
        self.turns.append(text)

        if resolve_references:
            image = self.image
            prompt = "Question: how many cats are there? Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            import pdb; pdb.set_trace()

    def write(self):
        pass
