#!/bin/bash

killall roscore
killall rosmaster
killall gzclient
killall gzserver
killall rviz

roscore -p $ROSPORT&

until rostopic list; do sleep 0.5; done #wait until rosmaster has started 

roslaunch classifier test.launch