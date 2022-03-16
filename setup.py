# -*- coding: utf-8 -*-
"""
Financial portfolio optimization with amplpy
--------------------------------------------

This package replicates some financial portfolio optimization models 
from `pypfopt <https://github.com/robertmartin8/PyPortfolioOpt>`_
using `amplpy <https://github.com/ampl/amplpy>`_.

Links
`````

* GitHub Repository: https://github.com/ampl/amplpyfinance
* PyPI Repository: https://pypi.python.org/pypi/amplpyfinance
"""
from setuptools import setup
import os


def ls_dir(base_dir):
    """List files recursively."""
    return [
        os.path.join(dirpath.replace(base_dir, "", 1), f)
        for (dirpath, dirnames, files) in os.walk(base_dir)
        for f in files
    ]


setup(
    name="amplpyfinance",
    version="0.0.0a0",
    description="Financial portfolio optimization with amplpy",
    long_description=__doc__,
    license="MIT",
    platforms="any",
    author="Filipe Brandão",
    author_email="fdabrandao@ampl.com",
    url="http://ampl.com/",
    download_url="https://github.com/ampl/amplpy/tree/master/amplpyfinance",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    install_requires=open("requirements.txt").read().split("\n"),
    packages=["amplpyfinance"],
    package_data={"": ls_dir("amplpyfinance/")},
)
