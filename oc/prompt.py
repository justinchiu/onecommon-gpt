import ast
import re
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)
from minichain import TemplatePrompt as BaseTemplatePrompt
from minichain import Output, Request, Prompt

from oc.outputs import UnderstandShortOutput
from oc.agent2.utils import Qtypes

import oc.prompts
from importlib.resources import files
PROMPT_DIR = files(oc.prompts)._paths[0]

HEADER = """from oc.fns.context import get_ctx
from oc.fns.shapes import is_triangle, is_line, is_square
from oc.fns.spatial import all_close, is_above, is_below, is_right, is_left, is_middle
from oc.fns.spatial import get_top, get_bottom, get_right, get_left
from oc.fns.spatial import get_top_right, get_top_left, get_bottom_right, get_bottom_left
from oc.fns.spatial import get_middle
from oc.fns.spatial import get_distance, get_minimum_radius
from oc.fns.color import is_dark, is_grey, is_light, lightest, darkest, same_color, different_color, is_darker, is_lighter
from oc.fns.size import is_large, is_small, is_medium_size, largest, smallest, same_size, different_size, is_larger, is_smaller
from oc.fns.iterators import get1idxs, get2idxs, get3idxs, getsets
from oc.fns.lists import add
from oc.fns.lists import sort_state
import numpy as np
from functools import partial
from itertools import permutations
"""

class TemplatePrompt(BaseTemplatePrompt[Output]):
    def print(self, kwargs):
        if self.template_file:
            tmp = Environment(loader=FileSystemLoader([".", "/"])).get_template(
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
    template_file = str(PROMPT_DIR / "reformat.j2")
    stop_templates = ["\n"]

class Parse(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "parse.j2")
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
    #template_file = str(PROMPT_DIR / "understand4.j2")
    template_file = str(PROMPT_DIR / "understand5.j2")
    stop_templates = ["# End.", "# New."]

    def parse(self, output, input):
        # debug
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
        print(len(encoding.encode(output)))
        # /debug
        return output


class Confirm(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "confirm.j2")
    stop_templates = ["# End."]

    def parse(self, output, input):
        word = output.strip()
        if word == "Yes":
            return True
        elif word == "No":
            return False
        elif word == "None":
            return None
        else:
            raise ValueError(f"Confirmation prompt returned: {word}")

class Execute(TemplatePrompt[list[int]]):
    #template_file = "prompts/execute.j2"
    #template_file = str(PROMPT_DIR / "execute3.j2")
    template_file = str(PROMPT_DIR / "execute5.j2")

    def parse(self, output, input) -> list[int]:
        return ast.literal_eval(output)

# Shortened prompts
class UnderstandShort(TemplatePrompt[str]):
    # input:
    #   * header: str
    #   * blocks: List[block]
    #   * speaker: str
    #   * text: str
    template_file = str(PROMPT_DIR / "understandshort.j2")
    stop_templates = ["# End."]

    def parse(self, output, input) -> UnderstandShortOutput | None:
        #print(output)
        #import pdb; pdb.set_trace()
        if "No op." in output:
            return None

        # debug
        print(output)
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
        print(len(encoding.encode(output)))
        # /debug

        code, dots, select, state = output.split("\n#")

        # remove last line from code
        code = code.strip()
        code = "\n".join(code.split("\n")[:-1])
        dots = dots.replace("Dots:", "").strip()
        select = select.replace("Selection:", "").strip()
        state = state.replace("State:", "").strip()

        if select == "False":
            # sometimes dots is incorrect, parse out the dots from the for loop
            for1, for2 = code.split("\n")[3:5]

            statedots = re.findall("for (.*?) in", for1)[0].replace(" ","").split(",")

            followupdots = re.findall("for (.*?) in", for2)[0].replace(" ","").split(",")
            #dots = statedots if "_" in followupdots else ",".join([statedots, followupdots])
            dots = ",".join(statedots + followupdots if "_" not in followupdots else statedots)

        # separate constraint names and assignment code
        constraint_lines = [line.strip() for line in code.split("\n") if "check" in line]
        constraint_pairs = [x.split(" = ") for x in constraint_lines]
        constraints = [dict(name=x[0], code=x[1]) for x in constraint_pairs]

        return UnderstandShortOutput(
            # skip the first line of code, which is the fn def
            code = "\n".join(code.split("\n")[1:]),
            constraints = constraints,
            dots = dots,
            select = select,
            state = state,
            speaker = input["speaker"],
            text = input["text"],
        )

class ExecuteShort(TemplatePrompt[list[int]]):
    template_file = str(PROMPT_DIR / "executeshort.j2")

    def parse(self, output, input) -> list[int]:
        return ast.literal_eval(output)

class Classify(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "classify.j2")
    template_file = str(PROMPT_DIR / "classify2.j2")
    stop_templates = ["End"]
    def parse(self, output, input) -> tuple[str, int, str]:
        output = output.strip()
        if output in [Qtypes.START.value, Qtypes.NOOP.value, Qtypes.FOLD.value, Qtypes.SELECT.value]:
            return output, 0, output
        else:
            qtype, num_dots = output.split("\n")
            num_dots = int(re.findall(r"\d+", num_dots)[0])
            return qtype, num_dots, output

class ClassifyZeroshot(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "classify_zs.j2")
    stop_templates = ["End"]
    def parse(self, output, input) -> tuple[str, int]:
        qtype, num_new = output.split("\n")
        qtype = int(re.findall(r"\d+", qtype)[0])
        num_new = int(re.findall(r"\d+", num_new)[0])
        if qtype == 1:
            qtype = Qtypes.START
        elif qtype == 2 and num_new == 0:
            qtype = Qtypes.FOLD
        elif qtype == 2 and num_new > 0:
            qtype = Qtypes.FNEW
        elif qtype == 3:
            qtype = Qtypes.NOOP
        elif qtype == 4:
            qtype = Qtypes.SELECT
        return qtype, num_new, output

class UnderstandShort2(TemplatePrompt[str]):
    # input:
    #   * header: str
    #   * blocks: List[block]
    #   * speaker: str
    #   * text: str
    template_file = str(PROMPT_DIR / "understandshort2.j2")
    stop_templates = ["```"]

    def parse(self, output, input) -> UnderstandShortOutput | None:
        # debug
        print(output)
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
        print(len(encoding.encode(output)))
        # /debug

        # separate constraint names and assignment code
        constraint_lines = output.split("\n")
        constraint_pairs = [x.split(" = ") for x in constraint_lines]
        constraints = [dict(name=x[0], code=x[1]) for x in constraint_pairs]
        return constraints

        return UnderstandShortOutput(
            # skip the first line of code, which is the fn def
            code = "\n".join(code.split("\n")[1:]),
            constraints = constraints,
            dots = dots,
            select = select,
            state = state,
            speaker = input["speaker"],
            text = input["text"],
        )

class ExecuteShort2(TemplatePrompt[list[int]]):
    template_file = str(PROMPT_DIR / "executeshort2.j2")

    def parse(self, output, input) -> list[int]:
        return ast.literal_eval(output)


# Deprecated
class ParseUnderstand(TemplatePrompt[str]):
    #template_file = "prompts/parseunderstand.j2"
    template_file = str(PROMPT_DIR / "parseunderstand3.j2")
    stop_templates = ["# End.", "# New."]

class Generate(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "generate.j2")

class GenerateScxy(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "generate_scxy.j2")


class UnderstandMc(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "understand_mc.j2")
    stop_template = '\n'

class GenerateTemplate(TemplatePrompt[str]):
    template_file = str(PROMPT_DIR / "generate_template.j2")

class GenerateMentions(TemplatePrompt[str]):
    tempalte_file = str(PROMPT_DIR / "generate_mention.j2")


# JSON shortened prompts
class UnderstandJson(TemplatePrompt[str]):
    # input:
    #   * header: str
    #   * blocks: List[block]
    #   * speaker: str
    #   * text: str
    template_file = str(PROMPT_DIR / "understandjson.j2")
    stop_templates = ["# End."]

    def parse(self, output, input):
        import pdb; pdb.set_trace()
        code, dots, selection = output.split("\n#")
        #code = code.strip()
        dots = dots.replace("Dots:", "").strip()
        selection = selection.replace("Selection:", "").strip()
        return code, dots, selection

class ExecuteJson(TemplatePrompt[list[int]]):
    template_file = str(PROMPT_DIR / "executejson.j2")

    def parse(self, output, input) -> list[int]:
        return ast.literal_eval(output)
