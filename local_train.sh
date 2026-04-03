if [ -n "$2" ] && [[ "$2" != --* ]]; then
    export CUDA_VISIBLE_DEVICES=$2
    extra_args=("${@:3}")
else
    export CUDA_VISIBLE_DEVICES=0
    extra_args=("${@:2}")
fi
export PYTHONPATH=$PYTHONPATH:./:./jrdb_toolkit/detection_eval

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}

config=projects/configs/$1.py

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=work_dirs/$1 \
        "${extra_args[@]}"
else
    python ./tools/train.py \
        ${config} \
        --work-dir=work_dirs/$1 \
        "${extra_args[@]}"
fi
