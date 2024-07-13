#! /bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: please enter these three arguments: dataset, config_path, ex_name."
    exit 1
fi

dataset=$1
config_path=$2
ex_name=$3

python3 tools/test.py $dataset $config_path $ex_name --mem_efficient_test
python3 tools/mem_efficient_test.py $ex_name
