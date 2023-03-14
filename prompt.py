from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)
from minichain import TemplatePrompt as McTemplatePrompt
from minichain import Output


class TemplatePrompt(McTemplatePrompt[Output]):
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


class Generate(TemplatePrompt[str]):
    template_file = "prompts/generate.j2"
