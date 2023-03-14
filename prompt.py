from minichain import TemplatePrompt


class Understand(TemplatePrompt[str]):
    template_file = "prompts/understand.j2"


class Generate(TemplatePrompt[str]):
    template_file = "prompts/generate.j2"
