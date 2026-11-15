Distributed Deep Learning Inference under Real-Time Constraint
==============

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

# Figure 6

```
Model_Name = {LeNet-Ciphar,LeNet-MNIST,MobileNet-Ciphar,MobileNet-MNIST,ResNet-Ciphar,ResNet-MNIST}

Directory_Name = {DeepThings, EdgeFlow, Proposed}
```

For each Model Name and Directory Name run the following:


```
cd <Directory_Name>\<Model_Name>

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
```
python workload.py
python plotheat.py
```

# Figure 8
```
python run_repeated.py
python plot_optimality_gap.py
python plot solver_time.py
```

# Figure 10

<img width="600" height="399" alt="2026-11-09T13_52_20 125061901" src="https://github.com/user-attachments/assets/62cdd366-d04c-416f-9b2c-4f72bc67d8c9" />


Terminal 1
```
gz sim -r -v 3 five_camera_warehouse.sdf
```
Terminal 2
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_script.py
```
Then to average the execution time per task run

```
python average.py
```
and plot the execution time by
```
python arrival_execution.py
```
