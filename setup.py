from setuptools import setup

setup(
    name = "oc",
    packages = ["oc", "tests"],
    include_package_data = True,
    package_data = {
        "oc": ["oc/prompts/*.j2"],
    },
)
