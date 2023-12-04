#!bin/bash

conda create -n ts_project_env python=3.9 -y -q
conda activate ts_project_env
conda install poetry
cd mlops_2023/ts_project
poetry install
