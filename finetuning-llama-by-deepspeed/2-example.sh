#!/bin/bash
WORKING_DIR=/opt/ml/code
SM_WORKING_DIR=/opt/ml/model


if [[ $NODE_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST=$SM_MASTER
    MASTER_ADDR=$SM_MASTER_ADDR
    MASTER_PORT="23456"
    NNODES="$NODE_NUMBER"
    NODE_RANK="$NODE_INDEX"
fi

GPUS_PER_NODE="$SM_NUM_GPUS"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SAVE_PATH="${SM_WORKING_DIR}/results"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/ds_z3_fp16.json"

EPOCHS=1


#model_id="decapoda-research/llama-7b-hf"
model_id="pinkmanlove/llama-7b-hf"

train_dataset_path='/opt/ml/input/data/train'
test_dataset_path='/opt/ml/input/data/test'
learning_rate=0.00001
model_max_length=1536
per_device_train_batch_size=1
per_device_eval_batch_size=1

OPTS=""
OPTS+=" --per_device_eval_batch_size ${per_device_eval_batch_size}"
OPTS+=" --per_device_train_batch_size ${per_device_train_batch_size}"
OPTS+=" --model_max_length ${model_max_length}"
OPTS+=" --model_name ${model_id}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --learning_rate ${learning_rate}"
OPTS+=" --training_dir ${train_dataset_path}"
OPTS+=" --test_dir ${test_dataset_path}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --num_train_epochs ${EPOCHS}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/train.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
