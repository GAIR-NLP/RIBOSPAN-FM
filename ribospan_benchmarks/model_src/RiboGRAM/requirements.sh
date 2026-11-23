#!/usr/bin/env bash
set -e

pip install --user torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --user matplotlib==3.7.5