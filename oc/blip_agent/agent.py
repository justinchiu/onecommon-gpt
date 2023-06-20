from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from oc.prompt import Reformat

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

    def feed_context(self, ctx, flip_y, belief_constructor=None):
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
