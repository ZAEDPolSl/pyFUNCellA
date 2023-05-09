# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["enrichment_auc"]

package_data = {"": ["*"]}

install_requires = [
    "matplotlib>=3.7.1,<3.8.0",
    "numpy-indexed>=0.3.7,<0.4.0",
    "numpy>=1.24.3,<1.25.0",
    "pandas>=2.0.1,<2.1.0",
    "scikit-learn>=1.2.2,<1.3.0",
    "scipy>=1.8,<1.9",
    "seaborn>=0.12.2",
    "tqdm>=4.65.0",
]

extras_require = {
    # ':python_version < "3.11"': ["importlib-metadata>=1.0,<2.0"],
    # "all": ["polyaxon>=1.5.0,<2.0.0", "gin-config>=0.5.0,<0.6.0"],
    # "gin": ["gin-config>=0.5.0,<0.6.0"],
    # "polyaxon": ["polyaxon>=1.5.0,<2.0.0"],
}


setup_kwargs = {
    "name": "enrichment_auc",
    "version": "0.1.0",
    "description": "GMM for single cell enrichment analysis",
    "long_description": "\n",
    "author": "Anna Mrukwa",
    "author_email": "a.mrukwa00@gmail.com",
    "maintainer": None,
    "maintainer_email": None,
    "url": "https://github.com/amrukwa/enrichment-auc",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "extras_require": extras_require,
    "python_requires": ">=3.6,<4.0",
}
from build import *

build(setup_kwargs)

setup(**setup_kwargs)
