#!/bin/bash
# scripts/train.sh
# 训练脚本

# 设置CUDA设备（如果有多个GPU，可以指定使用哪个）
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs
log_file=logs/train_$(date +%Y%m%d_%H%M%S).log
config_path=configs/FeedBack/config.yaml
mode=train

# 基础训练命令
nohup python main.py \
    --config $config_path \
    --mode $mode \
    > $log_file 2>&1 &

tail -f $log_file
