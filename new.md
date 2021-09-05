# NCR

[English](./README.md) | 简体中文

----
[![require](https://img.shields.io/badge/Paddle-2.1.2-brightgreen)](REQUIRE)

----

## 一、简介

本项目基于paddlepaddle框架复现 Neural Collaborative Reasoning，NCR基于神经网络构成的协同推理架构，使用了逻辑正则化算子来训练OR、NOT单元，以实现对用户-物品交互embedding的推理。


**论文:**
- [1] Chen, H. ,  Shi, S. ,  Y  Li, &  Y  Zhang. (2020). Neural collaborative reasoning.<br>

**参考项目：**
- [https://github.com/rutgerswiselab/NCR](https://github.com/rutgerswiselab/NCR)

**项目aistudio地址：**
- notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2289941](https://aistudio.baidu.com/aistudio/projectdetail/2289941)

## 二、复现精度

>该列指标在ML100k数据集上的评价指标

| |N@5|N@10|HR@5|HR@10|
| :---: | :---: | :---: | :---: | :---: |
|ML100k|0.3794|0.4369|0.5446|0.7208|


**预训练模型：**
[预训练权重](saved_model/NCR/0.3653_0.4254_0.5287_0.7144best_test.model)


## 三、环境依赖

- 硬件：GPU、CPU

- 框架：
    - PaddlePaddle = 2.1.2

## 五、快速开始

### Step1: clone

```bash
# clone this repo
git clone https://github.com/gsq7474741/Paddle-NCR
```
**安装依赖**
```bash
pip install -r requirements.txt
```

### Step2: 训练
```bash
python ./main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 1 --gpu 1
```

此时的输出为：
```
Test Before Training = 0.0243,0.0432,0.0393,0.0987 ndcg@5,ndcg@10,hit@5,hit@10
Prepare Train Data...
Prepare Validation Data...
Init: 	 train= 0.8473,0.8473 validation= 0.0276,0.0463,0.0393,0.0977 test= 0.0243,0.0432,0.0393,0.0987 [6.9 s] ndcg@5,ndcg@10,hit@5,hit@10
Optimizer: Adam
Epoch     1:   5%|██▌                                              | 22/416 [00:04<01:17,  5.07it/s]
```


### Step3: 评估&预测
```bash
python ./main.py --rank 1 --train 0 --load 1 --model_name NCR --model_path ../model/NCR/0.3653_0.4254_0.5287_0.7144best_test.model --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 1 --gpu 0
```
此时的输出为：
```
Test Before Training = 0.0432,0.0612,0.0732,0.1295 ndcg@5,ndcg@10,hit@5,hit@10
Load saved_model from saved_model/NCR/0.3653_0.4254_0.5287_0.7144best_test.model
Test After Training = 0.3794,0.4369,0.5446,0.7208 ndcg@5,ndcg@10,hit@5,hit@10
Save Test Results to result/result.npy
```
预测结果存储为result/result.npy（可添加命令行参数指定存储路径）
`

## 六、代码结构与详细说明

### 6.1 代码结构




```
.
├── README.md
├── configs
│   └── cfg.py
├── data_loaders
│   ├── DataLoader.py
├── data_processor
│   ├── DataProcessor.py
│   ├── HisDataProcessor.py
│   ├── ProLogicRecDP.py
├── dataset
│   ├── 5MoviesTV01-1-5
│   ├── Electronics01-1-5
│   ├── README.md
│   └── ml100k01-1-5
├── log
│   └── README.md
├── main.py
├── models
│   ├── BaseModel.py
│   ├── CompareModel.py
│   ├── NCR.py
│   ├── RecModel.py
├── new.md
├── readme_imgs
├── requirements.txt
├── result
│   ├── README.md
│   └── result.npy
├── run.sh
├── runners
│   ├── BaseRunner.py
│   ├── ProLogicRunner.py
├── saved_model
│   ├── NCR
│   │   ├── 0.3653_0.4254_0.5287_0.7144best_test.model
│   └── README.md
└── utils
    ├── dataset.py
    ├── mining.py
    ├── rank_metrics.py
    └── utils.py


```

```
├─config                          # 配置
├─dataset                         # 数据集加载
├─eval                            # 评估脚本
├─models                          # 模型
├─results                         # 可视化结果
├─utils                           # 工具代码
│  compile.sh                     # 编译pse.cpp
│  eval.py                        # 评估
│  init.sh                        # 安装依赖
│  predict.py                     # 预测
│  README.md                      # 英文readme
│  README_cn.md                   # 中文readme
│  requirement.txt                # 依赖
│  train.py                       # 训练
```

### 6.2 参数说明

可以在 `train.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  ----  |  ----  |  ----  |  ----  |
| config| None, 必选| 配置文件路径 ||
| --checkpoint| None, 可选 | 预训练模型参数路径 ||
| --resume| None, 可选 | 恢复训练 |例如：--resume checkpoint_44_0 不是断点参数的绝对路径请注意|


### 6.3 训练流程

#### 单机训练
```bash
python3 train.py $config_file
```

#### 多机训练
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py $config_file
```

此时，程序会将每个进程的输出log导入到`./debug`路径下：
```
.
├── debug
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
├── README.md
└── train.py
```

#### 训练输出
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。
```text
Epoch: [1 | 600]
(1/78) LR: 0.001000 | Batch: 6.458s | Total: 0min | ETA: 8min | Loss: 0.614 | Loss(text/kernel): 0.506/0.109 | IoU(text/kernel): 0.274/0.317 | Acc rec: 0.000
```

### 6.4 评估流程

```bash
python3 eval.py $config_file $pdparam_file --report_speed
```

此时的输出为：
```
Testing 1/500
backbone_time: 0.0266
neck_time: 0.0197
det_head_time: 0.0168
det_pse_time: 0.4697
FPS: 1.9
Testing 2/500
backbone_time: 0.0266
neck_time: 0.0197
det_head_time: 0.0171
det_pse_time: 0.4694
FPS: 1.9
Testing 3/500
backbone_time: 0.0266
neck_time: 0.0196
det_head_time: 0.0175
det_pse_time: 0.4691
FPS: 1.9
```

### 6.5 测试流程

```bash
python3 predict.py $config_file $input_dir $output_dir $pdparams_file --report_speed
```
此时的输出结果保存在`$output_dir`下面


### 6.6 使用预训练模型预测

使用预训练模型预测的流程如下：

**step1:** 下载预训练模型
谷歌云盘：[https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)
模型与配置文件对应关系：

|name|path|config|
| :---: | :---: | :---: |
|pretrain_1|psenet_r50_ic17_1024_Adam/checkpoint_33_0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_2|psenet_r50_ic17_1024_Adam/checkpoint_46_0|[psenet_r50_ic17_1024.py](./config/psenet/psenst_r50_ic17_1024.py)|
|pretrain_3|psenet_r50_ic17_1260_Adam/checkpoint_68_0|[psenet_r50_ic17_1260.py](./config/psenet/psenet_r50_ic17_1260.py)|
|ic15_finetune_1|psenet_r50_ic15_1024_Adam/checkpoint_491_0|[finetune1.py](./config/psenet/finetune1.py)|
|ic15_finetune_2|psenet_r50_ic15_1260_Adam/best|[finetune2.py](./config/psenet/finetune2.py)|
|ic15_finetune_3|psenet_r50_ic15_1480_SGD/checkpoint_401_0|[finetune3.py](./config/psenet/finetune3.py)|
|tt_finetune_1|psenet_r50_tt/checkpoint_331_0|[psenet_r50_tt.py](./config/psenet/psenet_r50_tt.py)|
|tt_finetune_2|psenet_r50_tt/checkpoint_290_0|[psenet_r50_tt_finetune2.py](./config/psenet/psenet_r50_tt_finetune2.py)|

**step2:** 使用预训练模型完成预测
```bash
python3 predict.py $config_file $input_dir $output_dir $pdparams_file --report_speed
```
## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 徐铭远、衣容颉|
| 时间 | 2021.05 |
| 框架版本 | Paddle 2.0.2 |
| 应用场景 | 文本检测 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)  |
| 在线运行 | [botebook](https://aistudio.baidu.com/aistudio/projectdetail/1945560)、[脚本任务](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1796445)|
