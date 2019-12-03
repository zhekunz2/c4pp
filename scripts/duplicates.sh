#!/usr/bin/env bash
fdupes -r -f progs2019* | grep ctemplate | xargs -I{} sh -c "mv {} {}.dup"