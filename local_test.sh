export PYTHONPATH=$PYTHONPATH:./:./jrdb_toolkit/detection_eval
export PORT=29532

if [ -n "$3" ] && [[ "$3" != --* ]]; then
    export CUDA_VISIBLE_DEVICES=$3
    extra_args=("${@:4}")
else
    export CUDA_VISIBLE_DEVICES=0
    extra_args=("${@:3}")
fi

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}

config=projects/configs/$1.py
checkpoint=$2

echo "number of gpus: "${gpu_num}
echo "config file: "${config}
echo "checkpoint: "${checkpoint}

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_test.sh \
        ${config} \
        ${checkpoint} \
        ${gpu_num} \
        --eval bbox \
        "${extra_args[@]}"
else
    python ./tools/test.py \
        ${config} \
        ${checkpoint} \
        --eval bbox \
        "${extra_args[@]}"
fi
