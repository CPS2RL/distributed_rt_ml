# Validation record

- Six model/dataset C inference cores pass `gcc -std=c11 -Wall -Wextra -Werror -fsyntax-only` with ESP timer mock headers.
- Shared ESP-NOW transport `main/main.c` passes the same host C11 syntax checks for the MNIST configuration; CIFAR-10 is checked separately.
- Numerical validation: C local-halo inference compared with PyTorch global full-image features (using a test transport that supplies exact needed remote rows), all three device partitions, randomly initialized test models with BatchNorm folding:
  * LeNet5 MNIST: max absolute logit difference 3.02e-8
  * ResNet20 MNIST: 4.52e-8
  * MobileNetV1 0.25 MNIST: 7.49e-8
  * LeNet5 CIFAR-10: 4.47e-8
  * ResNet20 CIFAR-10: 8.94e-8
  * MobileNetV1 0.25 CIFAR-10: 6.71e-8
- The transport itself uses requested-row GET/FRAGMENT and SET/ACK packets plus per-layer readiness/DONE barriers. These exchanges have **not** been tested across physical ESP32 devices, nor under ESP-IDF linking.
- The numeric test substitutes exact reference halo rows; it is not a hardware radio or latency test.
