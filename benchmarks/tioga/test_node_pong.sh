#!/bin/bash

cd /g/g14/bienz1/BenchPress/tioga_build/examples
flux mini run --setop=mpibind=off -n 128 --verbose --exclusive --nodes=2 ./time_node_pong

