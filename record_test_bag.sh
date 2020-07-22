#!/bin/bash 

rosbag record -o /home/bag/Myhal_Simulation/viz_bags/ /tf /tf_static /classified_points /velodyne_points /gmapping_points2 /local_planner_points2 /amcl_points2 /global_planner_points2 /clock
