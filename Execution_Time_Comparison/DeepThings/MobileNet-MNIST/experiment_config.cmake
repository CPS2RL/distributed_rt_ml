# Select one of six trained CNNs. Reconfigure, build and flash same firmware to all boards.
set(EXPERIMENT_MODEL "mobilenetv1_025")     # lenet5 | resnet20 | mobilenetv1_025
set(EXPERIMENT_DATASET "mnist")    # mnist | cifar10
set(EXPERIMENT_RUNS 50)            # 1..250
