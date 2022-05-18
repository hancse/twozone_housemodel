# https://realpython.com/pypi-publish-python-package/


import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="han-housemodel",
    version="0.1.0",
    description="HAN lumped element house model with appliances and control",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hancse/han-housemodel",
    author="HAN-AEA-BES",
    author_email="info@realpython.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=("tests",)),
    # packages=["housemodel"],
    include_package_data=True,
    install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
        ]
    },
)

# https://www.freecodecamp.org/news/how-to-create-and-upload-your-first-python-package-to-pypi/

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()
    
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
