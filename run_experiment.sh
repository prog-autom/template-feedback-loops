#!/bin/bash

trap 'exit' SIGINT SIGTERM SIGHUP SIGQUIT

if [ -z "$1" ]
then
  printf "Iterates the feedback loop experiment over many parameters configs using the grid search.\n"
  printf "Parameters inside this scripts are configurable: step, usage, adherence."
  printf "\n\n"
  printf "Usage: run_experiment.sh <pipeline_name>\n\n"
  printf "See ./experiment.yaml for more details\n"
  exit
fi

for step in 10 20
do
    for usage in 0.{1..9}
    do
        for adherence in 0.{1..9}
        do
          export step &&
          export usage &&
          export adherence &&
          mldev --config .mldev/config.yaml run -f ./experiment.yml --no-commit $1
        done
    done
done