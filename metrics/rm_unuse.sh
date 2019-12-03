#!/usr/bin/env bash
param_use=$1
tt=$(echo $(grep + $param_use) | sed "s/+/\"/g" | sed "s/ /[^a-zA-Z_\.]\\\|/g")
if [ ! -z "$tt" ]; then grep "${tt}[^a-zA-Z_\.]" $2; fi
