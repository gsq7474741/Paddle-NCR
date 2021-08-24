# NCR-paddle

[![require](https://img.shields.io/badge/Paddle-2.1.2-brightgreen)](REQUIRE)

## 简介

* 使用 Paddle 框架复现论文 Neural Collaborative Reasoning
* 复现指标： ML100k:HR@10>0.68
* Test After Training = 0.6847 hr@10
* [原文地址](https://arxiv.org/pdf/2005.08129.pdf)

## 论文摘要

现有的协同过滤（CF）方法大多是基于匹配的思想设计的，然而，作为一项认知而非感知的智能任务，推荐不仅需要数据的模式识别和匹配能力，还需要数据的认知推理能力。 本文中作者提出了协同推理（CR），提出了一个模块化的推理体系结构，学习且(∧) ,
或(∨) , 非（¬）等逻辑符号作为神经网络来实现蕴涵推理( →) 。

网络结构如下

[![img](https://github.com/gsq7474741/Paddle-NCR/blob/main/readme_imgs/image2.png)](IMG)

网络的embedding和逻辑单元均使用双层全连接，使用ReLU作为激活函数

[![img](https://github.com/gsq7474741/Paddle-NCR/blob/main/readme_imgs/image3.png)](IMG)

作者使用逻辑正则化来训练或（OR）、非（NOT）模块

[![img](https://github.com/gsq7474741/Paddle-NCR/blob/main/readme_imgs/image6.png)](IMG)

## 快速使用

### 环境

* Python>=3.7

* 具体要求见见[requirements.txt](https://github.com/gsq7474741/Paddle-NCR/blob/main/requirements.txt)

* ```requirements.txt
  paddlepaddle==2.1.2
  numpy==1.18.1
  pandas==0.24.2
  scipy==1.3.0
  tqdm==4.32.1
  scikit_learn==0.23.1
  ```

### 模型训练

* 使用命令行启动模型训练

  ```shell
  $ cd src
  $ python ./main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 2022 --gpu 1
  ```
* 训练日志记录在/log下

### 模型测试

* 使用命令行测试模型

```shell
  $ cd src
  $ python ./main.py --rank 1 --train 0 --load 1 --model_name NCR --model_path ../model/NCR/pre2.model --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 2022 --gpu 0
   ```