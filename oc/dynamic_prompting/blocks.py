from pathlib import Path
import jinja2
import json

from importlib.resources import files
import oc.promptdata
PROMPT_DATA_DIR = str(files(oc.promptdata)._paths[0])
import oc.prompts
PROMPT_DIR = str(files(oc.prompts)._paths[0])

dir = Path(PROMPT_DATA_DIR)
blockfile = dir / "blocks-temp.json"
with blockfile.open("r") as f:
    BLOCKS = json.load(f)


def codeblock(kwargs):
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPT_DIR))
    template = environment.get_template("codeblock.j2")
    return template.render(**kwargs)

def codeblocks():
    return [codeblock(block) for block in BLOCKS]


if __name__ == "__main__":
    blocks = codeblocks()
    import pdb; pdb.set_trace()
