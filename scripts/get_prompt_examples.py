from enum import Enum
from pathlib import Path
import re
from oc.dynamic_prompting.blocks import BLOCKS

from oc.dynamic_prompting.construct import question_type, constraints, constraints_no_var, constraints_dots
strings = question_type()
#print(strings)
with Path("scratch/short-example.txt").open("w") as f:
    f.write("\n\n".join(strings))


strings = constraints()
with Path("scratch/short-code-example.txt").open("w") as f:
    f.write("\n\n".join(strings))

strings = constraints_no_var()
with Path("scratch/short-code-novar-example.txt").open("w") as f:
    f.write("\n\n".join(strings))

strings = constraints_dots()
with Path("scratch/short-code-dots-example.txt").open("w") as f:
    f.write("\n\n".join(strings))
