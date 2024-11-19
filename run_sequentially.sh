#!/bin/bash


#for i in {6..30}
#do
    # Construct the argument for --path
  #  path_arg="assets/socket/insert_socket_$i"
 #   
    # Run the python script with the current path argument
   # python3 video_to_gripper_hamer_kpts.py --task_folder $path_arg
#done


for i in {0..30}
do
    # Construct the argument for --path
    path_arg="../assets/demos/kettle/kettle_on_stove_$i"
    
    # Run the python script with the current path argument
    python3 video_to_gripper_hamer_kpts.py --task_folder $path_arg
done


