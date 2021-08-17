# NCR-paddle
## 简介
* 使用 Paddle 框架复现论文 Neural Collaborative Reasoning
* 复现指标： ML100k:HR@10>0.68
* Test After Training = 0.6847 hr@10
* [原文地址](https://arxiv.org/pdf/2005.08129.pdf)

## 快速使用

### 模型训练
* 使用命令行启动模型训练

  ```shell
  $ cd src
  $ python ./main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 2022 --gpu 1
  ```
* 训练日志记录在/log下
