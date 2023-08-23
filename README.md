# onecommon-gpt

## Installation
```
pip install -e .
```

## Selfplay
After installation, clone the repo `justinchiu/onecommon` and run `bash onecommon/aaai2020/experiments/jc_run_gpt_selfplay.sh`.

## Round-trip
Run `python scripts/generation_playground2.py`.

## Reference resolution
Run `python oc/eval/eval.py --run_refres --model gpt-4 --num_examples 25`
for the reference resolution evaluation.

Generation evaluation has been deprecated, as we covering the empirical distribution
of utterances is not necessary for success.

## Examples for creating prompts
* `python get_prompting_examples.py` has examples for generation with features.

## Visualization
`streamlit run dot.py` visualizes one reference resolution with codegen example.
`streamlit run dot.py --split 0 --method parsecodegen --model gpt-4`
