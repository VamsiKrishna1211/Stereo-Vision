# Stereo-Vision

## For Kinova arm ros2 kortex
Followed this GitHub: https://github.com/Kinovarobotics/ros2_kortex <br />
This video at times (just enters the same commands listed in the GitHub above): https://www.youtube.com/watch?v=Vcb_A1MmC-g <br />
Resolved error from this issue post: https://github.com/ros-controls/ros2_control/issues/1889 <br />

### Creating Docker container
Assuming we need a Docker container (system is not Ubuntu 22.04):
```
sudo rocker --x11 --name humble --nocleanup osrf/ros:humble-desktop
```
### Within Docker container
```
apt install git -y
```
```
apt install python3-colcon-common-extensions python3-vcstool
```
```
export COLCON_WS=~/workspace/ros2_kortex_ws
mkdir -p $COLCON_WS/src
```
```
cd $COLCON_WS
git clone https://github.com/Kinovarobotics/ros2_kortex.git src/ros2_kortex
vcs import src --skip-existing --input src/ros2_kortex/ros2_kortex.$ROS_DISTRO.repos
vcs import src --skip-existing --input src/ros2_kortex/ros2_kortex-not-released.$ROS_DISTRO.repos
```
```
vcs import src --skip-existing --input src/ros2_kortex/simulation.humble.repos
```
IF realtime_tools IS STILL **NOT** SYNCED TO NEWEST VERSION ON humble BRANCH. Last time I ran everything this was the case and had to do the following step.
```
cd $COLCON_WS
git clone https://github.com/ros-controls/realtime_tools.git ~/workspace/ros2_kortex_ws/src/realtime_tools
apt update
rosdep install --ignore-src --from-paths src -y -r
```
IF realtime_tools **IS** SYNCED TO NEWEST VERSION ON humble BRANCH. Only run this if you didn't already run the command in the previous step.
```
rosdep install --ignore-src --from-paths src -y -r
```
Colcon build may fail/freeze your system!!! Below, the parallel workers flag is already trying to fix this. If this still freezes, then use the command below that sets the execution to sequential (it will take a lot more time but shouldn't freeze)
```
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers 3
source ~/workspace/ros2_kortex_ws/install/setup.bash
```
Sequential execution for colcon build if needed:
```
export MAKEFLAGS="-j 1"
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --executor sequential
source ~/workspace/ros2_kortex_ws/install/setup.bash
```
Everything should be set up! Test:
```
ros2 launch kortex_description view_robot.launch.py gripper:=robotiq_2f_85 dof:=6
```


## Publishing joint trajectory
In one terminal:
```
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  robot_type:=gen3 \
  dof:=6 \
  gripper:=robotiq_2f_85 \
  use_sim_time:=true
```

In another terminal:
```
ros2 control load_controller joint_trajectory_controller
```
```
ros2 control set_controller_state joint_trajectory_controller inactive
```
```
ros2 control set_controller_state joint_trajectory_controller active
```
```
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6],
  points: [
    { positions: [0, 0, 0, 0, 0, 0], time_from_start: { sec: 10 } },
  ]
}" -1
```
To control gripper (in an additional terminal), `0.0=open`, `0.8=close`
```
ros2 control load_controller robotiq_gripper_controller
```
```
ros2 control set_controller_state robotiq_gripper_controller inactive
```
```
ros2 control set_controller_state robotiq_gripper_controller active
```
```
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command:{position: 0.0, max_effort: 100.0}}"
```
This lists the controllers (good for debugging)
```
ros2 control list_controllers
```
