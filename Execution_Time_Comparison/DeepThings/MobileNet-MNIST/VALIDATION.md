# Validation record — NO-HALO WITH ONE-TIME OVERLAPPED INPUT AND REDUNDANT COMPUTATION

All six sources were compiled with `gcc -std=c11 -Wall -Wextra -Werror -fsyntax-only` using a stub for the ESP-IDF timer; all six model/dataset combinations executed successfully with host-based synthetic model weights on all three device IDs. These tests do not prove numerical equivalence with real trained checkpoints, ESP-IDF compatibility, memory sufficiency on hardware, or ESP-NOW reliability.

The host tests use a small synthetic image and synthetic deterministic float weights. The release does not include pretrained weights.
