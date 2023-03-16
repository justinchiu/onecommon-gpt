import ast
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)
from minichain import TemplatePrompt as BaseTemplatePrompt
from minichain import Output, Request, Prompt

HEADER = """from dot import get_dots
from shapes import is_triangle, is_line, is_square
from spatial import all_close, is_close, is_above, is_below, is_right, is_left
from spatial import get_top, get_bottom, get_right, get_left, get_top_right, get_top_left, get_bottom_right, get_bottom_left
from color import is_dark, is_grey, is_light
from size import is_large, is_small, largest, smallest, is_medium
from iterators import get2dots, get3dots
import numpy as np
"""

class TemplatePrompt(BaseTemplatePrompt[Output]):
    def print(self, kwargs):
        if self.template_file:
            tmp = Environment(loader=FileSystemLoader(".")).get_template(
                name=self.template_file
            )
        elif self.template:
            tmp = self.template  # type: ignore
        else:
            tmp = Template(self.prompt_template)
        if isinstance(kwargs, dict):
            x = tmp.render(**kwargs)
        else:
            x = tmp.render(**asdict(kwargs))
        return x


class Understand(TemplatePrompt[str]):
    template_file = "prompts/understand.j2"
    stop_template = "#"


class Execute(TemplatePrompt[list[int]]):
    template_file = "prompts/execute.j2"

    def parse(self, output, input) -> list[int]:
        #print(output)
        #import pdb; pdb.set_trace()
        return ast.literal_eval(output)


class Generate(TemplatePrompt[str]):
    template_file = "prompts/generate.j2"


class UnderstandMc(TemplatePrompt[str]):
    template_file = "prompts/understand_mc.j2"

class GenerateTemplate(TemplatePrompt[str]):
    template_file = "prompts/generate_template.j2"
