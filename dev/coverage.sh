#!/bin/bash
cd "`dirname "$0"`"
cd ..
set -ex

# python -m pip install coverage
coverage erase
coverage run --omit */site-packages/* -a -m amplpyfinance.tests
coverage report
coverage html
