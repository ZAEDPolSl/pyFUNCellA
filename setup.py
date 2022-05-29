# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['enrichment-auc']

package_data = \
{'': ['*']}

install_requires = \
['joblib>=1.0.0,<2.0.0',
 'matplotlib>=3.3.3,<4.0.0',
 'numpy-indexed>=0.3.5,<0.4.0',
 'numpy>=0.12.1',
 'pandas>=0.20.3',
 'scikit-learn>=0.19.0',
 'scipy>=0.19.1',
 'seaborn>=0.11.2,<0.12.0',
 'tqdm>=4.11.2']

extras_require = \
{':python_version < "3.8"': ['importlib-metadata>=1.0,<2.0'],
 'all': ['polyaxon>=1.5.0,<2.0.0', 'gin-config>=0.5.0,<0.6.0'],
 'gin': ['gin-config>=0.5.0,<0.6.0'],
 'polyaxon': ['polyaxon>=1.5.0,<2.0.0']}


setup_kwargs = {
    'name': 'enrichment-auc',
    'version': '3.0.12',
    'description': 'GMM for single cell enrichment analysis',
    'long_description': '\n',
    'author': 'Anna Mrukwa',
    'author_email': 'a.mrukwa00@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/amrukwa/enrichment-auc',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.6,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
