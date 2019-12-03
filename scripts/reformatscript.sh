#!/usr/bin/env bash
find ~/projects/c4pp/programs/bugs_examples/ -name "script.txt" | xargs -I{} sh -c "./getInits.sh {}"
#find ~/Desktop/research/c4pp/programs/bugs_examples/ -name "script.txt" | xargs -I{} sh -c "./getInits.sh {}"
