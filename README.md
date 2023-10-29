# onecommon-gpt

## Installation
```
pip install -e .
```

## Prompts

Full prompt
* Template: oc-anon/oc/prompts/understand5.j2
* Link: https://pastebin.com/x8dxj4tz

SPC prompts

Reformat: Rewrite partner utterances to correct typos and formatting.
* Template: oc-anon/oc/prompts/reformat.j2
* Example: https://pastebin.com/uTeP8nZE

Classify: Classify the dialogue act and any reference to previous turns.
* Template: oc-anon/oc/prompts/classify2.j2
* Example: https://pastebin.com/FFY8APRX

Confirm: Classify whether the partner said yes or no to a previous question.
* Template: oc-anon/oc/prompts/confirm.j2
* Example: https://pastebin.com/YpTiTkSD

Constraint generation: Generate the constraints.
* Template: oc-anon/oc/prompts/understandshort2.j2
* Example: https://pastebin.com/ube83h3n

Execution template: Take the output of previous prompts and turn them into a program.
* Template: oc-anon/oc/prompts/executeshort2.j2
* Example: https://pastebin.com/JJvGAtyU

GPT4 w/o codegen prompt
* System prompt: https://pastebin.com/8k1tVn4w
* Example: https://pastebin.com/tq9S2UtU

## Selfplay
After installation, clone the repo `justinchiu/onecommon` and run `bash onecommon/aaai2020/experiments/jc_run_gpt_selfplay.sh`.

## Round-trip
Run `python scripts/generation_playground2.py`.
