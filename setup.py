"""Install topopt."""

from setuptools import setup
import os

import topopt

here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
long_description = long_description.replace(
    "assets/imgs",
    "https://raw.githubusercontent.com/zfergus/topopt/master/assets/imgs/"
)

setup(
    name=topopt.__name__,
    packages=[topopt.__name__],
    version=topopt.__version__,
    license=topopt.__license__,
    description="A Python Library for Topology Optimization",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=topopt.__author__,
    author_email=topopt.__email__,
    url="https://github.com/zfergus/topopt",
    keywords=["Topology Optimization",
              "Sturctural Optimization", "Simulation"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">= 3.5",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "nlopt",
        "cvxopt",
        "pathlib; python_version < '3.4'",
    ],
    project_urls={
        "Bug Reports": "https://github.com/zfergus/topopt/issues",
        "Source": "https://github.com/zfergus/topopt/",
    },
)
