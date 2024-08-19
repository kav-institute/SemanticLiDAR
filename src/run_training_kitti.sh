#!/bin/bash

# Set common parameters
LEARNING_RATE=0.001
NUM_EPOCHS=30
BATCH_SIZE=8
NUM_WORKERS=16
SCRIPT_PATH="/home/appuser/repos/train_semantic_KITTI.py"

# Specific parameters for certain models
SMALL_BATCH_SIZE=4
SMALL_NUM_WORKERS=8

# Array of model types
MODEL_TYPES=(
    #'resnet50'
    #'regnet_y_1_6gf'
    #'regnet_y_3_2gf'
    #'shufflenet_v2_x1_5'
    'resnet34'
    #'regnet_y_800mf'
    #'shufflenet_v2_x1_0'
    #'resnet18'
    #'regnet_y_400mf'
    #'shufflenet_v2_x0_5'
)

# Loop through each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
    if [[ "$MODEL_TYPE" == "resnet50" || "$MODEL_TYPE" == "regnet_y_3_2gf"  || "$MODEL_TYPE" == "shufflenet_v2_x1_5" || "$MODEL_TYPE" == "shufflenet_v2_x1_0" ]]; then
        BATCH_SIZE=$SMALL_BATCH_SIZE
        NUM_WORKERS=$SMALL_NUM_WORKERS
    else
        BATCH_SIZE=8
        NUM_WORKERS=16
    fi

    # Loop through combinations of --attention and --normals flags
    for ATTENTION_FLAG in "--attention" ""
    do
        for NORMALS_FLAG in "--normals" ""
        do
            echo "Training with model: $MODEL_TYPE, Batch size: $BATCH_SIZE, Num workers: $NUM_WORKERS, Attention: $ATTENTION_FLAG, Normals: $NORMALS_FLAG"
            python $SCRIPT_PATH --model_type $MODEL_TYPE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --rotate --flip $ATTENTION_FLAG $NORMALS_FLAG
        done
    done
done
