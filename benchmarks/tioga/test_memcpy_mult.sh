#!/bin/bash

cd /g/g14/bienz1/BenchPress/tioga_build/examples
flux mini run --setop=mpibind=off -n 64 --verbose --exclusive --nodes=1 ./time_memcpy_mult

