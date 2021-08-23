#!/bin/bash
#加载环境，此处加载 anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/7.6.5.32_cuda10.2
module load nccl/2.9.6-1_cuda10.2
export LD_LIBRARY_PATH=/data/apps/cudnn/7.6.5.32_cuda10.2/cuda/lib64:${LD_LIBRARY_PATH}
/data/apps/cudnn/7.6.5.32_cuda10.2/cuda/lib64
#true
export LD_LIBRARY_PATH=/data/apps/cuda/10.2/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3
#conda create -n pdlt python=3.8 -y
conda init bash
conda activate pdlt
#pip install paddlepaddle
#pip install x2paddle
#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
#cd ./Paddle-NCR/src || exit
python ./src/main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 2022 --gpu 1
