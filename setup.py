import sys

import setuptools

sys.path.insert(0, "src")
import agc_optims

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="agc_optims",
    version=agc_optims.__version__,
    author="Fabian Deuser",
    description="Easy to use optimizers in with adaptive gradient clipping. Written in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skyy93/agc_optims",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
    install_requires=[
        "torch >= 1.6.0",
    ],
)