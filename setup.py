from setuptools import setup, find_packages

setup(
    name = "oc",
    #packages = ["oc", "tests"],
    packages = find_packages(),
    include_package_data = True,
    package_data = {
        "oc.data": [
            "oc/data/onecommon/train_reference_1.txt",
            "data/onecommon/train_reference_1.txt",
        ],
        "oc.prompts": ["oc/prompts/*.j2"],
    },
)
