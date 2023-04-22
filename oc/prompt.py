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

import oc.prompts
from importlib.resources import files
PROMPT_DIR = files(oc.prompts)._paths[0]

HEADER = """from context import get_ctx
from shapes import is_triangle, is_line, is_square
from spatial import all_close, is_above, is_below, is_right, is_left, is_middle
from spatial import get_top, get_bottom, get_right, get_left
from spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from spatial import get_middle
from spatial import get_distance, get_minimum_radius
from color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, is_darker, is_lighter
from size import is_large, is_small, is_medium_size, largest, smallest, same_size, different_size, is_larger, is_smaller
from iterators import get1idxs, get2idxs, get3idxs, getsets
from lists import add
from lists import sort_state
import numpy as np
from functools import partial
from itertools import permutations
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
    #template_file = "prompts/reformat.j2"
    template_file = PROMPT_DIR / "reformat.j2"
    stop_templates = ["\n"]

class Parse(TemplatePrompt[str]):
    template_file = PROMPT_DIR / "parse.j2"
    stop_templates = ["\nEnd."]

    def parse(self, output, input):
        outs = output.split("\n")
        confirmation = outs[0]
        description = outs[2:] if len(outs) > 2 else None
        selection = "selection" in output
        return output, confirmation, description, selection

class Understand(TemplatePrompt[str]):
    #template_file = "prompts/understand.j2"
    #template_file = "prompts/understand2.j2"
    #template_file = "prompts/understand3.j2"
    template_file = PROMPT_DIR / "understand4.j2"
    stop_templates = ["# End.", "# New."]

class ParseUnderstand(TemplatePrompt[str]):
    #template_file = "prompts/parseunderstand.j2"
    template_file = PROMPT_DIR / "parseunderstand3.j2"
    stop_templates = ["# End.", "# New."]

class Execute(TemplatePrompt[list[int]]):
    #template_file = "prompts/execute.j2"
    template_file = PROMPT_DIR / "execute3.j2"

    def parse(self, output, input) -> list[int]:
        return ast.literal_eval(output)


class Generate(TemplatePrompt[str]):
    template_file = PROMPT_DIR / "generate.j2"

class GenerateScxy(TemplatePrompt[str]):
    template_file = PROMPT_DIR / "generate_scxy.j2"


class UnderstandMc(TemplatePrompt[str]):
    template_file = PROMPT_DIR / "understand_mc.j2"
    stop_template = '\n'

class GenerateTemplate(TemplatePrompt[str]):
    template_file = PROMPT_DIR / "generate_template.j2"

class GenerateMentions(TemplatePrompt[str]):
    tempalte_file = PROMPT_DIR / "generate_mention.j2"
