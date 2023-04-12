#!/bin/bash
cd "`dirname "$0"`"
cd ..
set -ex

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <version>"
else
  version=$1
  sed -i~ "s/amplpyfinance==.*/amplpyfinance==$version/" docs/requirements.txt
  sed -i~ "s/version=\"[^']*\"/version=\"$version\"/" setup.py
  sed -i~ "s/__version__ = \"[^']*\"/__version__ = '$version'/" amplpyfinance/__init__.py
fi
