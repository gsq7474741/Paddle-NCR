#  #encoding=utf8
#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from main import main
import paddle
import paddle.distributed as dist

argdict = {'train': 1, 'load': 0,
           'dataset': "'ml100k01-1-5'", 'metric': "'ndcg@5,ndcg@10,hit@5,hit@10'", 'max_his': 5, 'test_neg_n': 100,
           'l2': 1e-4,
           'r_weight': 0.1, 'random_seed': 1, 'gpu': '"0"'}

if paddle.device.get_device() is "cpu":
    main(argdict)
else:
    dist.spawn(main, args=(argdict))
