#!/bin/bash
pkill roscore
pkill rosmaster
pkill gzclient
pkill gzserver
pkill rviz

until rostopic list; do sleep 0.5; done #wait until rosmaster has started

roslaunch classifier online_frame_preds.launch
