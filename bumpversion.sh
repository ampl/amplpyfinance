#!/bin/bash
set -ex
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <version>"
else
  version=$1
  sed -i~ "s/version=\"[^']*\"/version=\"$version\"/" setup.py
  sed -i~ "s/__version__ = \"[^']*\"/__version__ = '$version'/" amplpyfinance/__init__.py
fi
