#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

import setuptools

from cmake.cmake_extension import (
    BuildExtension,
    bdist_wheel,
    cmake_extension,
    is_windows,
)


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


def get_package_version():
    with open("CMakeLists.txt") as f:
        content = f.read()

    match = re.search(r"set\(KHG_VERSION (.*)\)", content)
    latest_version = match.group(1).strip('"')
    return latest_version


package_name = "kaldi-hmm-gmm"

with open("kaldi-hmm-gmm/python/kaldi_hmm_gmm/__init__.py", "a") as f:
    f.write(f"__version__ = '{get_package_version()}'\n")

install_requires = [
    "graphviz",
    "kaldialign",
    "kaldifst",
    "numpy",
    "kaldilm",
    # please install torch separately
]

setuptools.setup(
    name=package_name,
    python_requires=">=3.6",
    install_requires=install_requires,
    version=get_package_version(),
    author="The next-gen Kaldi development team",
    author_email="csukuangfj@gmail.com",
    package_dir={
        "kaldi_hmm_gmm": "kaldi-hmm-gmm/python/kaldi_hmm_gmm",
    },
    packages=["kaldi_hmm_gmm"],
    url="https://github.com/csukuangfj/kaldi-hmm-gmm",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[cmake_extension("_kaldi_hmm_gmm")],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel},
    zip_safe=False,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache licensed, as found in the LICENSE file",
)

with open("kaldi-hmm-gmm/python/kaldi_hmm_gmm/__init__.py", "r") as f:
    lines = f.readlines()

with open("kaldi-hmm-gmm/python/kaldi_hmm_gmm/__init__.py", "w") as f:
    for line in lines:
        if "__version__" in line:
            # skip __version__ = "x.x.x"
            continue
        f.write(line)
