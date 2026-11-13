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
Terminal 1
```
gz sim -r -v 3 five_camera_warehouse.sdf
```
Terminal 2
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python five_camera_sorter.py
```

