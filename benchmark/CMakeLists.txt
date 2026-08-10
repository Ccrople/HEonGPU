# Copyright 2024-2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/benchmark)

set(EXECUTABLES
    bfv_benchmark benchmark_bfv.cpp
    ckks_benchmark benchmark_ckks.cpp
    tfhe_benchmark benchmark_tfhe.cpp
    ckks_profile profile_ckks.cpp
    llama3_profile profile_llama3.cpp
    llama3_batch_profile profile_llama3_batch.cpp
    llama3_nobatch_profile profile_llama3_nobatch.cpp
    llama3_rect_profile profile_llama3_rect.cpp
    llama3_rect_boot_profile profile_llama3_rect_boot.cpp
    llama3_levels_profile profile_llama3_levels.cpp
    boot_precision_sweep profile_boot_precision.cpp
    ring_cost_profile profile_ring_cost.cpp
    boot_to_slots_profile profile_boot_to_slots.cpp
    slot_island_profile profile_slot_island.cpp
    attention_layout_profile profile_attention_layout.cpp
    low_ring_params_profile profile_low_ring_params.cpp
    ringswitch_profile profile_ringswitch.cpp
    tworing_attention_profile profile_tworing_attention.cpp
)

function(add_benchmark exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart)
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
endfunction()

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR EXECUTABLES_COUNT "${EXECUTABLES_LENGTH} / 2")
math(EXPR EXECUTABLES_COUNT_LOOP "${EXECUTABLES_COUNT} - 1")

foreach(i RANGE 0 ${EXECUTABLES_COUNT_LOOP})
    math(EXPR index1 "${i} * 2")
    math(EXPR index2 "${i} * 2 + 1")
    list(GET EXECUTABLES ${index1} exe)
    list(GET EXECUTABLES ${index2} source)
    add_benchmark(${exe} ${source})
endforeach()
