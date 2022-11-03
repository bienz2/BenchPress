#!/bin/bash

cd /g/g14/bienz1/BenchPress/tioga_build/examples
flux mini run -N1 -n1 -c8 -g8 ./time_memcpy
