GPU=0
DATASET=upmc_food101_cmml
TYPE=both
RATIO=0.7
EXP_NOTE="${DATASET}_${TYPE}_${RATIO}"
#!/bin/bash
#SBATCH ... (这里是您的sbatch指令)

echo "--- 我现在在Slurm作业环境内部 ---"
echo "我的作业ID是: $SLURM_JOB_ID"
echo "我所在的服务器是: $(hostname)"
echo "Slurm告诉我的CUDA_VISIBLE_DEVICES是: '$CUDA_VISIBLE_DEVICES'"
echo "--- 准备启动Python脚本 ---"

TOKENIZERS_PARALLELISM=True python src/main.py experiment=rebq_mixlora_${DATASET} data=${DATASET} EXP_NOTE=${EXP_NOTE} train.GPU=${GPU} data.missing_params.RATIO=${RATIO} data.missing_params.TYPE=${TYPE} train.EVAL_FREQ=1000 data.NUM_WORKERS=2 test.BATCH_SIZE=4
# TOKENIZERS_PARALLELISM=True python src/main.py experiment=rebq_mixlora_${DATASET} data=${DATASET} EXP_NOTE=${EXP_NOTE} train.GPU=${GPU} data.missing_params.RATIO=${RATIO} data.missing_params.TYPE=${TYPE} train.EVAL_FREQ=1000 data.NUM_WORKERS=8 test.BATCH_SIZE=4 test.TEST_ONLY=True test.CHECKPOINT_DIR=checkpoints/${EXP_NOTE}/checkpoints/

# GPU=0
# DATASET=upmc_food101_cmml
# TYPE=both
# RATIO=0.7
# # 更新：在实验笔记中加入新模型的名字以作区分
# EXP_NOTE="${DATASET}_rebq_mixlora_${TYPE}_${RATIO}"

# # 训练阶段
# # 更新：将 experiment 参数指向我们为新模型创建的配置文件
# TOKENIZERS_PARALLELISM=True python src/main.py \
#     experiment=rebq_mixlora_${DATASET} \
#     EXP_NOTE=${EXP_NOTE} \
#     GPU=${GPU} \
#     data.missing_params.RATIO=${RATIO} \
#     data.missing_params.TYPE=${TYPE} \
#     data.NUM_WORKERS=8 

# # 测试阶段
# # 更新：确保 CHECKPOINT_DIR 指向正确的、由新实验笔记生成的路径
# TOKENIZERS_PARALLELISM=True python src/main.py \
#     experiment=rebq_mixlora_${DATASET} \
#     EXP_NOTE=${EXP_NOTE} \
#     GPU=${GPU} \
#     data.missing_params.RATIO=${RATIO} \
#     data.missing_params.TYPE=${TYPE} \
#     data.NUM_WORKERS=8 \
#     test.TEST_ONLY=True \
#     test.CHECKPOINT_DIR=checkpoints/${EXP_NOTE}/checkpoints/

