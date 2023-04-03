# onecommon-gpt

## Dependencies
* justinchiu/MiniChain fork, chatgpt branch

## Evaluation
Run `python eval.py --run_gen --run_refres --refres codegen --gen feat`
for the latest.

## Examples for creating prompts
* `python get_prompting_examples.py` has examples for generation with features.

## Visualization
`streamlit run dot.py` visualizes one reference resolution with codegen example.
`streamlit run dot.py --split 0 --method parsecodegen --model gpt-4`
