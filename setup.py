# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages

setup(
    name='taccl',
    version='2.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'taccl = taccl.__main__:main',
        ],
    },
    install_requires=[
        'dataclasses; python_version < "3.7"',
        'z3-solver',
        'argcomplete',
        'lxml',
        'gurobipy',
        'numpy'
    ],
    python_requires='>=3.6',
)
