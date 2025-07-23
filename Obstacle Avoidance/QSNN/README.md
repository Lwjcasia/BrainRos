# Spiking Neural Network for Mapless Navigation with QSNN #

This package provides the PyTorch implementation of the **Q**-learning based **S**piking **N**eural **N**etwork (**QSNN**) framework for energy-efficient mapless navigation.

## Software Installation ##

#### 1. Basic Requirements

* Ubuntu 16.04
* Python 3.5.2
* ROS Kinetic (with Gazebo 7.0)
* PyTorch 1.2 (with CUDA 10.0 and tensorboard 2.1)

ROS Kinetic is not compatible with Python 3 by default. If you have issues using Python 3 with ROS, please follow this [link](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674) to resolve them. We use the default Python 2 environment to execute `roslaunch` and `rosrun`.

A CUDA-enabled GPU is not required but is preferred for training.

We have provided `requirements.txt` for the Python environment. We recommend setting up the environment using [virtualenv](https://pypi.org/project/virtualenv/).

#### 2. Simulation Setup

The simulation environment uses a Turtlebot2 robot with a 360-degree LiDAR in Gazebo.

Install Turtlebot2 dependencies:
```bash
sudo apt-get install ros-kinetic-turtlebot-*
```

We use the Hokuyo LiDAR model. Install its dependencies:
```bash
sudo apt-get install ros-kinetic-urg-node
```

Download the project and compile the catkin workspace:
```bash
cd <Dir>/<Project Name>/ros/catkin_ws
catkin_make
```

Add the following lines to your `~/.bashrc` to set up the ROS environment properly:
```bash
source <Dir>/<Project Name>/ros/catkin_ws/devel/setup.bash
export TURTLEBOT_3D_SENSOR="hokuyo"
```

Run `source ~/.bashrc` and test the environment by running (in a Python 2 environment):
```bash
roslaunch turtlebot_lidar turtlebot_world.launch
```
You should see the Turtlebot2 with a LiDAR on top.

## Example Usage ##

#### 1. Training QSNN ####

First, launch the training world, which includes 4 different environments (use a Python 2 environment and an absolute path for `<Dir>`):
```bash
roslaunch turtlebot_lidar turtlebot_world.launch world_file:=<Dir>/<Project Name>/ros/worlds/training_worlds.world
```

Then, run the `laserscan_simple` ROS node in a separate terminal to sample laser scan data every 10 degrees (use a Python 2 environment):
```bash
rosrun simple_laserscan laserscan_simple
```

Now, start the training in a new terminal (use a Python 3 environment):
```bash
source <Dir to Python 3 Virtual Env>/bin/activate
cd <Dir>/<Project Name>/training/train_spiking_qsnn
python train_qsnn.py --cuda 1 --step 5
```
This will train the model for 1000 episodes and save the trained parameters every 10k steps. Intermediate results are logged via TensorBoard. To train on a CPU, set `--cuda` to 0. You can adjust the inference timesteps by changing the `--step` value.

#### 2. Evaluate in Simulated Environment ####

To evaluate the trained Spiking Actor Network (SAN) in Gazebo, first launch the evaluation world (use a Python 2 environment and an absolute path for `<Dir>`):
```bash
roslaunch turtlebot_lidar turtlebot_world.launch world_file:=<Dir>/<Project Name>/ros/worlds/evaluation_world.world
```

Then, run the `laserscan_simple` ROS node in a separate terminal (use a Python 2 environment):
```bash
rosrun simple_laserscan laserscan_simple
```

Now, start the evaluation in a new terminal (use a Python 3 environment):
```bash
source <Dir to Python 3 Virtual Env>/bin/activate
cd <Dir>/<Project Name>/evaluation/eval_random_simulation
python run_qsnn_eval.py --save 0 --cuda 1 --step 5
```
This will navigate the robot through 200 randomly generated start and goal positions. To evaluate on a CPU, set `--cuda` to 0. You can also change the inference timesteps with the `--step` argument.

Set `--save` to 1 to save robot routes and time. To analyze the saved history, run:
```bash
source <Dir to Python 3 Virtual Env>/bin/activate
cd <Dir>/<Project Name>/evaluation/result_analyze
python generate_results.py
```
