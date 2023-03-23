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

HEADER = """from context import get_ctx
from shapes import is_triangle, is_line, is_square
from spatial import all_close, are_above, are_below, are_right, are_left
from spatial import are_above_left, are_above_right, are_below_right, are_below_left
from spatial import are_middle
from spatial import get_top, get_bottom, get_right, get_left
from spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from spatial import get_middle
from color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, are_darker, are_lighter
from size import is_large, is_small, is_medium, largest, smallest, same_size, different_size, are_larger, are_smaller
from iterators import get1idxs, get2idxs, get3idxs
from lists import add
import numpy as np
from functools import partial
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

class Reformat(TemplatePrompt[str]):
    template_file = "prompts/reformat.j2"
    stop_templates = ["\n"]

class Understand(TemplatePrompt[str]):
    #template_file = "prompts/understand.j2"
    template_file = "prompts/understand2.j2"
    stop_templates = ["# End.", "# New."]

class Execute(TemplatePrompt[list[int]]):
    template_file = "prompts/execute.j2"

    def parse(self, output, input) -> list[int]:
        #print(output)
        #import pdb; pdb.set_trace()
        return ast.literal_eval(output)


class Generate(TemplatePrompt[str]):
    template_file = "prompts/generate.j2"

class GenerateScxy(TemplatePrompt[str]):
    template_file = "prompts/generate_scxy.j2"


class UnderstandMc(TemplatePrompt[str]):
    template_file = "prompts/understand_mc.j2"
    stop_template = '\n'

class GenerateTemplate(TemplatePrompt[str]):
    template_file = "prompts/generate_template.j2"

class GenerateMentions(TemplatePrompt[str]):
    tempalte_file = "prompts/generate_mention.j2"
