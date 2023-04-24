from setuptools import setup

setup(
    name = "oc",
    packages = ["oc", "tests"],
    include_package_data = True,
    package_data = {
        "oc.data": [
            "oc/data/onecommon/train_reference_1.txt",
            "data/onecommon/train_reference_1.txt",
        ],
        "oc.prompts": ["oc/prompts/*.j2"],
    },
)
