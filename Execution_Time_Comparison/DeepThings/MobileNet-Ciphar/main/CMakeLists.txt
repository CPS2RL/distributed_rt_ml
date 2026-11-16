include("${CMAKE_CURRENT_LIST_DIR}/../experiment_config.cmake")
set(MODEL_OPTIONS lenet5 resnet20 mobilenetv1_025)
set(DATASET_OPTIONS mnist cifar10)
if(NOT EXPERIMENT_MODEL IN_LIST MODEL_OPTIONS)
    message(FATAL_ERROR "Invalid EXPERIMENT_MODEL: ${EXPERIMENT_MODEL}")
endif()
if(NOT EXPERIMENT_DATASET IN_LIST DATASET_OPTIONS)
    message(FATAL_ERROR "Invalid EXPERIMENT_DATASET: ${EXPERIMENT_DATASET}")
endif()
if(NOT EXPERIMENT_RUNS MATCHES "^[0-9]+$")
    message(FATAL_ERROR "EXPERIMENT_RUNS must be an integer")
endif()
if(EXPERIMENT_RUNS LESS 1 OR EXPERIMENT_RUNS GREATER 250)
    message(FATAL_ERROR "EXPERIMENT_RUNS must be 1..250")
endif()

set(MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}/../variants/${EXPERIMENT_MODEL}_${EXPERIMENT_DATASET}")
if(EXPERIMENT_MODEL STREQUAL "lenet5")
    set(MODEL_ID 1)
    set(KERNEL_SOURCE "distributed_lenet.c")
elseif(EXPERIMENT_MODEL STREQUAL "resnet20")
    set(MODEL_ID 2)
    set(KERNEL_SOURCE "distributed_resnet.c")
else()
    set(MODEL_ID 3)
    set(KERNEL_SOURCE "distributed_mobilenet.c")
endif()
if(EXPERIMENT_DATASET STREQUAL "mnist")
    set(DATASET_ID 1)
    set(INPUT_WIDTH 28)
    set(INPUT_CHANNELS 1)
else()
    set(DATASET_ID 2)
    set(INPUT_WIDTH 32)
    set(INPUT_CHANNELS 3)
endif()

idf_component_register(
    SRCS "main.c" "model_adapter.c"
         "${MODULE_DIR}/${KERNEL_SOURCE}"
         "${MODULE_DIR}/model_weights.c"
    INCLUDE_DIRS "." "${MODULE_DIR}"
    REQUIRES esp_wifi esp_event esp_netif nvs_flash esp_hw_support esp_timer
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    EXP_MODEL_ID=${MODEL_ID} EXP_DATASET_ID=${DATASET_ID}
    EXP_RUNS=${EXPERIMENT_RUNS}
    EXP_IMG_W=${INPUT_WIDTH} EXP_IMG_C=${INPUT_CHANNELS}
    MOBILE_IMG_W=${INPUT_WIDTH} MOBILE_IMG_C=${INPUT_CHANNELS}
)
message(STATUS "Distributed CNN selection: ${EXPERIMENT_MODEL} / ${EXPERIMENT_DATASET}, runs=${EXPERIMENT_RUNS}")
