#! /usr/bin/env bash

#/progs*/progs*/
ls -d progs*/progs*/ | tail -5 | xargs -P12 -n1 -I{} sh -c "echo {}; ./pre_run_metrics.sh \$(realpath cmdstan-2.16.0) {} > {}/metrics_out"
#ls -d progs*/progs*/ | tail -5 | xargs -P6 -n1 -I{} sh -c "cd {}; ../../cmdstan-2.16.0/bin/stansummary "
