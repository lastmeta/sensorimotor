import os
from setuptools import setup, find_packages

def get_long_description():
    with open("README.md", "r") as fh:
        return fh.read()

def get_name():
    return 'sensorimotor'

def get_version():
    return '0.0.1'

setup(
    name=get_name(),
    version=get_version(),
    description='development of a sensorimotor inference engine; a distributed collaboration framework',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=[f'{get_name()}.{p}' for p in find_packages(where=get_name())],
    install_requires=[],
    python_requires='>=3.5.2',
    author='Jordan Miller',
    author_email="paradoxlabs@protonmail.com",
    url="https://github.com/propername/maestro",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # scripts=[f for f in findall(dir='maestro/bin') if f.endswith('.py')],
    entry_points={
        "console_scripts": [
            "sm = sensorimotor.cli.sensorimotor:main",
        ]
    },
)
