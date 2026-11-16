Distributed Deep Learning Inference under Real-Time Constraint
==============

https://anonymous.4open.science/r/distributed_rt_ml-7C5F/

##### Table of Contents  
- [Execution Time Comparison](#figure-6)  
- [Workload Distribution](#figure-7) 
- [Algorithm Comparison](#figure-8) 
- [Warehouse Simulation](#figure-10) 

# Environment Setup
## ESP-IDF 
### Installation (Windows)
Download and install ESP-IDF Installation Manager from https://dl.espressif.com/dl/eim/
### Connect the ESP32 Devices
Open the IDF PowerShell and run the following command to list the COM ports of the attached devices
```
python -m serial.tools.list_ports
```

> COM9
> COM10
> COM11
> COM12

### Build and Flash the Application
Build the project and flash the application onto the three devices:
```
cd ~\Figure_6\DeepThings\LeNet-MNIST
idf.py set-target esp32
idf.py build

idf.py -p COM10 flash
idf.py -p COM11 flash
idf.py -p COM9 flash
```
Monitor the output from the designated device:
```
idf.py -p COM9 monitor
```
Press `Ctrl+]` to exit the serial monitor.

## Gazebo Installation
Install Gazebo Harmonic (LTS) on top of Ubuntu 24.04.
Add the Gazebo Repository
```
sudo apt update && sudo apt install software-properties-common
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update
sudo apt install -y gz-harmonic
gz sim --versions
```

### Verify the Warehouse Simulation
```
cd ~/Figure_10
gz sim -r -v 3 five_camera_warehouse.sdf
```

## Python Virtual Environment
```
sudo apt install -y python3 python3-pip python3-venv

python3 -m venv .venv
```

# Figure 6

This experiment compares the execution time of the three CNN inference methods (DeepThings, EdgeFlow Proposed) covering three types of architecture (LeNet, MobileNet and ResNet) trained using two types of dataset (using the MNIST and CIFAR-10). The goal of this experiment is to evaluate the execution time of different CNN architectures for different types of workload.

## Experimental Configurations

- Methods: DeepThings, EdgeFlow, Proposed
- Models: LeNet5, MobileNetV1-0.25, ResNet20
- Datasets: MNIST, CIFAR-10
- Hardware: ESP32 devices

Each method is evaluated using six combinations which covers 18 experimental configurations.

## Evaluation

Open IDF PowerShell and navigate to the desired method and model directory.

> Model_Name = {LeNet-Ciphar,LeNet-MNIST,MobileNet-Ciphar,MobileNet-MNIST,ResNet-Ciphar,ResNet-MNIST}

> Directory_Name = {DeepThings, EdgeFlow, Proposed}

```
cd <Directory_Name>\<Model_Name>
```

For example,

```
cd DeepThings\LeNet-Ciphar
```
## Build and Flash the Application

For each Model Name and Directory Name run the following:


```
idp.py build
idf.py COM10 flash
idf.py COM11 flash
idf.py COM9 flash
idf.py COM9 monitor
```

Save the output as a log file plot the execution time. Or instead, run the plotting codes
```
python plot_cumulative-<Directory_Name>-<Model_Name>.py
```

# Figure 7

This evaluation covers the workload distribution of the CNN inference methods. The workload distribution is visualized using heatmaps.  Each cell represents the computational workload assigned to a particular device at a specific network layer. An uniform distribution indicates better load balancing.

```
python workload.py
python plotheat.py
```

The result presents a layer-by-layer visualization of the computational workload distribution across ten devices. 

# Figure 8

This evaluation compares the solution quality and computational overhead of the proposed method and a heuristic method. The experiment focuses on two evaluation metrics: optimality gap and solver execution time. A smaller optimality gap indicates that the heuristic algorithm produces a solution closer to the proposed method.

```
python run_repeated.py
python plot_optimality_gap.py
python plot solver_time.py
```

# Figure 10

In this evaluation, we have simulated a warehouse environment. We used Gazebo to simulate a warehouse with five cameras. In this evaluation we analyzed the execution time of inference tasks.

<img width="600" height="399" alt="2026-11-09T13_52_20 125061901" src="https://github.com/user-attachments/assets/62cdd366-d04c-416f-9b2c-4f72bc67d8c9" />


## Launch the Warehouse Simulation
In terminal 1, run the following: 
```
gz sim -r -v 3 five_camera_warehouse.sdf
```

## Execute the Inference Experiment
In terminal 2, run the following:
```
source .venv/bin/activate

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_script.py
```
After completing the inference experiment, calculate the average execution time for each task by running:

```
python average.py
```
To visualize the relationship between task arrival times and execution times, run:
```
python arrival_execution.py
```
