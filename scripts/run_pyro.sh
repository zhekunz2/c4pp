#!/usr/bin/env bash
# usage: ./scripts/run_pyro.sh
cd programs/templates
# run all programs
echo "Running..."
ls | xargs -n 1 -P 10 -I{} sh -c "echo {}; cd {}; timeout 10m python {}.py > output 2>&1; echo \"{} done\""