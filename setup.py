from setuptools import setup, find_packages

setup(
    name = "oc",
    packages = [
        "oc",
        "oc.agent",
        "oc.agent2",
        "oc.belief",
        "oc.eval",
        "oc.fns",
        "oc.gen",
        "oc.dynamic_prompting",
        "tests",
    ],
    include_package_data = True,
    package_data = {
        "oc.data": [
            "oc/data/onecommon/train_reference_1.txt",
            "oc/data/onecommon/valid_reference_1.txt",
        ],
        "oc.prompts": ["oc/prompts/*.j2"],
        "oc.promptdata": ["oc/promptdata/*"],
    },
    install_requires = [
        "datasets",
        "evaluate",
        "openai",
        "scikit-learn",
        "shapely",
        "tenacity",
        "seaborn",
        "more-itertools",
        "torch",
        "MiniChain @ git+https://github.com/justinchiu/MiniChain@chatgpt#egg=minichain",
    ],
)
