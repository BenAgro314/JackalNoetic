#!/bin/bash
pkill roscore
pkill rosmaster
pkill gzclient
pkill gzserver
pkill rviz

roslaunch classifier classifier.launch