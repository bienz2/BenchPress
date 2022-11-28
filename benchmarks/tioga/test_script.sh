#!/bin/bash

cd /g/g14/bienz1/BenchPress/tioga_build/examples

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo $ROCR_VISIBLE_DEVICES

./time_ping_pong_gpu
