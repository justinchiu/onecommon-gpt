from setuptools import setup, find_packages

setup(
    name = "oc",
    packages = [
        "oc",
        "oc.agent",
        "oc.belief",
        "oc.eval",
        "oc.fns",
        "oc.gen",
        "tests",
    ],
    include_package_data = True,
    package_data = {
        "oc.data": [
            "oc/data/onecommon/train_reference_1.txt",
            "data/onecommon/train_reference_1.txt",
        ],
        "oc.prompts": ["oc/prompts/*.j2"],
    },
)
