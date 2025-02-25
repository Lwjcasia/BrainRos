#! /bin/bash
source /home/v/nav_ws/devel/setup.bash
roslaunch my_robot_name_2dnav move_base.launch & roslaunch obstacle obst.launch

