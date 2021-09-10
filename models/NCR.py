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

# from utils.utils import concat
import numpy as np
import paddle
import paddle.nn.functional as F

from configs import cfg
from models.BaseModel import BaseModel
from utils import utils


class NCR(BaseModel):
    append_id = True
    include_id = False
    include_user_features = False
    include_item_features = False

    @staticmethod
    def parse_model_args(parser, model_name='NCR'):
        parser.add_argument('--u_vector_size', type=int, default=64, help= \
            'Size of user vectors.')
        parser.add_argument('--i_vector_size', type=int, default=64, help= \
            'Size of item vectors.')
        parser.add_argument('--r_weight', type=float, default=10, help= \
            'Weight of logic regularizer loss')
        parser.add_argument('--ppl_weight', type=float, default=0, help= \
            'Weight of uv interaction prediction loss')
        parser.add_argument('--pos_weight', type=float, default=0, help= \
            'Weight of positive purchase loss')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, label_min, label_max, feature_num, user_num,
                 item_num, u_vector_size, i_vector_size, r_weight, ppl_weight,
                 pos_weight, random_seed, model_path):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num
        self.r_weight = r_weight
        self.ppl_weight = ppl_weight
        self.pos_weight = pos_weight
        self.sim_scale = 10
        BaseModel.__init__(self, label_min=label_min, label_max=label_max,
                           feature_num=feature_num, random_seed=random_seed, model_path= model_path)

    def _init_weights(self):
        self.iid_embeddings = paddle.nn.Embedding(self.item_num, self.ui_vector_size)
        self.iid_embeddings.weight.stop_gradient = False
        self.uid_embeddings = paddle.nn.Embedding(self.user_num, self.ui_vector_size)
        self.uid_embeddings.weight.stop_gradient = False
        self.true = paddle.create_parameter(
            shape=utils.numpy_to_torch(np.random.uniform(0, 0.1, size=self.ui_vector_size).astype(np.float32)).shape,
            dtype=str(utils.numpy_to_torch(
                np.random.uniform(0, 0.1, size=self.ui_vector_size).astype(np.float32)).numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(
                utils.numpy_to_torch(np.random.uniform(0, 0.1, size=self.ui_vector_size).astype(np.float32))))

        self.true.stop_gradient = False

        self.not_layer_1 = paddle.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.not_layer_1.weight.stop_gradient = False
        self.not_layer_2 = paddle.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.not_layer_2.weight.stop_gradient = False
        self.and_layer_1 = paddle.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.and_layer_1.weight.stop_gradient = False
        self.and_layer_2 = paddle.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.and_layer_2.weight.stop_gradient = False
        self.or_layer_1 = paddle.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.or_layer_1.weight.stop_gradient = False
        self.or_layer_2 = paddle.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.or_layer_2.weight.stop_gradient = False
        self.purchase_layer_1 = paddle.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_1.weight.stop_gradient = False
        self.purchase_layer_2 = paddle.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_2.weight.stop_gradient = False

    def logic_not(self, vector):
        vector = F.relu(self.not_layer_1(vector))
        vector = self.not_layer_2(vector)
        return vector

    def logic_and(self, vector1, vector2):
        assert len(vector1.shape) == len(vector2.shape)
        vector = paddle.concat((vector1, vector2), axis=len(vector1.shape) - 1)
        vector = F.relu(self.and_layer_1(vector))
        vector = self.and_layer_2(vector)
        return vector

    def logic_or(self, vector1, vector2):
        assert len(vector1.shape) == len(vector2.shape)
        vector = paddle.concat((vector1, vector2), axis=len(vector1.shape) - 1)
        vector = F.relu(self.or_layer_1(vector))
        vector = self.or_layer_2(vector)
        return vector

    def purchase_gate(self, uv_vector):
        uv_vector = F.relu(self.purchase_layer_1(uv_vector))
        uv_vector = self.purchase_layer_2(uv_vector)
        return uv_vector

    def mse(self, vector1, vector2):
        return paddle.mean(((vector1 - vector2) ** 2))

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]
        history = feed_dict[cfg.C_HISTORY]
        batch_size, his_length = history.shape
        history_pos_tag = paddle.cast(paddle.unsqueeze(feed_dict[cfg.C_HISTORY_POS_TAG], 2), dtype='float32')
        user_vectors = self.uid_embeddings(u_ids)
        item_vectors = self.iid_embeddings(i_ids)
        item_vectors = paddle.concat((user_vectors, item_vectors), axis=1)
        item_vectors = self.purchase_gate(item_vectors)
        uh_vectors = paddle.reshape(user_vectors, [user_vectors.shape[0], 1, user_vectors.shape[1]])
        uh_vectors = paddle.expand(uh_vectors,
                                   [history_pos_tag.shape[0], history_pos_tag.shape[1], uh_vectors.shape[2]])
        his_vectors = self.iid_embeddings(history)
        his_vectors = paddle.concat((uh_vectors, his_vectors), axis=2)
        his_vectors = self.purchase_gate(his_vectors)
        not_his_vectors = self.logic_not(his_vectors)
        constraint = list([his_vectors])
        constraint.append(not_his_vectors)
        his_vectors = history_pos_tag * his_vectors + (1 - history_pos_tag) * not_his_vectors
        tmp_vector = self.logic_not(his_vectors[:, 0])
        shuffled_history_idx = [i for i in range(1, his_length)]
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, self.logic_not(his_vectors[:, i]))
            constraint.append(paddle.reshape(tmp_vector, [batch_size, -1, self.ui_vector_size]))
        left_vector = tmp_vector
        right_vector = item_vectors
        constraint.append(paddle.reshape(right_vector, [batch_size, -1, self.ui_vector_size]))
        sent_vector = self.logic_or(left_vector, right_vector)
        constraint.append(paddle.reshape(sent_vector, [batch_size, -1, self.ui_vector_size]))
        if feed_dict['rank'] == 1:
            prediction = F.cosine_similarity(sent_vector, paddle.reshape(self.true, [1, -1])) * 10
        else:
            prediction = paddle.nn.functional.cosine_similarity(sent_vector, paddle.reshape(self.true, [1, -1])) * (
                    self.label_max - self.label_min
            ) / 2 + (self.label_max + self.label_min) / 2
        constraint = paddle.concat(constraint, axis=1)
        out_dict = {'prediction': prediction, 'check': check_list,
                    'constraint': constraint, 'interim': left_vector}
        return out_dict

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        check_list = out_dict['check']
        false = paddle.reshape(self.logic_not(self.true), [1, -1])
        constraint = out_dict['constraint']
        dim = len(constraint.shape) - 1
        r_not_not_true = paddle.sum(
            (1 - paddle.nn.functional.cosine_similarity(self.logic_not(self.logic_not(self.true)), self.true, axis=0)))
        r_not_not_self = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_not(self.logic_not(constraint)), constraint,
                                                       axis=dim))
        r_not_self = paddle.mean(
            1 + paddle.nn.functional.cosine_similarity(self.logic_not(constraint), constraint, axis=dim))
        r_not_self = paddle.mean(
            1 + paddle.nn.functional.cosine_similarity(self.logic_not(constraint), constraint, axis=dim))
        r_not_not_not = paddle.mean(
            1 + paddle.nn.functional.cosine_similarity(self.logic_not(self.logic_not(constraint)),
                                                       self.logic_not(constraint), axis=dim))
        r_and_true = paddle.mean(1 - paddle.nn.functional.cosine_similarity(
            self.logic_and(constraint, paddle.expand_as(self.true, constraint)), constraint, axis=dim))
        r_and_false = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_and(constraint, paddle.expand_as(false, constraint)),
                                                       paddle.expand_as(false, constraint), axis=dim))
        r_and_self = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_and(constraint, constraint), constraint, axis=dim))
        r_and_not_self = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_and(constraint, self.logic_not(constraint)),
                                                       paddle.expand_as(false, constraint), axis=dim))
        r_and_not_self_inverse = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_and(self.logic_not(constraint), constraint),
                                                       paddle.expand_as(false, constraint), axis=dim))
        r_or_true = paddle.mean(1 - paddle.nn.functional.cosine_similarity(
            self.logic_or(constraint, paddle.expand_as(self.true, constraint)), paddle.expand_as(self.true, constraint),
            axis=dim))
        r_or_false = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_or(constraint, paddle.expand_as(false, constraint)),
                                                       constraint, axis=dim))
        r_or_self = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_or(constraint, constraint), constraint, axis=dim))
        r_or_not_self = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_or(constraint, self.logic_not(constraint)),
                                                       paddle.expand_as(self.true, constraint), axis=dim))
        r_or_not_self_inverse = paddle.mean(
            1 - paddle.nn.functional.cosine_similarity(self.logic_or(self.logic_not(constraint), constraint),
                                                       paddle.expand_as(self.true, constraint), axis=dim))
        true_false = 1 + paddle.nn.functional.cosine_similarity(self.true, paddle.reshape(false, [-1]), axis=0)
        r_loss = (r_not_not_true + r_not_not_self + r_not_self + r_and_true +
                  r_and_false + r_and_self + r_and_not_self +
                  r_and_not_self_inverse + r_or_true + r_or_false + r_or_self +
                  true_false + r_or_not_self + r_or_not_self_inverse + r_not_not_not)
        r_loss = r_loss * self.r_weight
        if feed_dict['rank'] == 1:
            batch_size = int(feed_dict['Y'].shape[0] / 2)
            pos, neg = out_dict['prediction'][:batch_size], out_dict[
                                                                'prediction'][batch_size:]
            loss = -paddle.sum(paddle.log(F.sigmoid((pos - neg))))
        else:
            loss = paddle.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])
        loss = loss + r_loss
        # loss.stop_gradient=True
        out_dict['loss'] = loss
        out_dict['check'] = check_list
        return out_dict
