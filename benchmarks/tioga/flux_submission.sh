#!/bin/bash 
flux mini batch --exclusive -o mpibind=off -N 1 -n 1 -c 8 ./test_memcpy.sh
