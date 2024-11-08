#!/bin/bash

rundir=$(date +"run-$(hostname -s)-%Y%m%d-%H%M%S")
mkdir -p $rundir

function run() {
    echo config=$config
    echo args=$@

    python3 run_flux.py --steps 4  "$@" > >(tee $rundir/stdout-s4-$config.log)  2> >(tee $rundir/stderr-s4-$config.log)
    python3 run_flux.py --steps 25 "$@" > >(tee $rundir/stdout-s25-$config.log) 2> >(tee $rundir/stderr-s25-$config.log)
    python3 run_flux.py --steps 50 "$@" > >(tee $rundir/stdout-s50-$config.log) 2> >(tee $rundir/stderr-s50-$config.log)

    if [ $? -eq 0 ]; then
        nsys profile --cuda-memory-usage true -o $rundir/report-$config.nsys-rep python3 run_flux.py --steps 4 "$@"
    fi
}

config=bf16-compile
run --config bf16 --compile

config=bf16-t5-compile
run --config bf16-t5 --compile

config=int8dq-compile
run --config bf16 --torchao --compile

config=int8dq-t5-compile
run --config bf16-t5 --torchao --compile

config=int8dq-nocompile
run --config bf16 --torchao 

config=int8dq-t5-nocompile
run --config bf16-t5 --torchao 

for cfg in svdq svdq-t5 w4a4 w4a4-t5 bf16 bf16-t5 nf4 nf4-t5; do
    config=$cfg
    run --config $cfg

    config=$cfg-ol1
    run --config $cfg --offload 1

    config=$cfg-ol2
    run --config $cfg --offload 2
done