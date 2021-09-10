#  #encoding=utf8

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

# from x2paddle import torch2paddle
import copy
import itertools
import logging
import os
import pickle
from time import time

import numpy as np
import paddle
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from configs import cfg
from utils import utils
from utils.utils import Momentum, Adam, clip_grad_value_


class BaseRunner(object):

    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load saved_model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=0.0001,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='GD',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default='RMSE',
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--skip_eval', type=int, default=0,
                            help='number of epochs without evaluation')
        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100,
                 batch_size=128, eval_batch_size=128 * 128, dropout=0.2, l2=1e-05,
                 metrics='RMSE', check_epoch=10, early_stop=1):
        """
        初始化
        :param optimizer: 优化器名字
        :param learning_rate: 学习率
        :param epoch: 总共跑几轮
        :param batch_size: 训练batch大小
        :param eval_batch_size: 测试batch大小
        :param dropout: dropout比例
        :param l2: l2权重
        :param metrics: 评价指标，逗号分隔
        :param check_epoch: 每几轮输出check一次模型中间的一些tensor
        :param early_stop: 是否自动提前终止训练
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None
        self.train_results, self.valid_results, self.test_results = [], [], []

    def _build_optimizer(self, model):
        """
        创建优化器
        :param model: 模型
        :return: 优化器
        """
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info('Optimizer: GD')
            optimizer = Momentum(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
        elif optimizer_name == 'adam':
            logging.info('Optimizer: Adam')
            optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
        else:
            logging.error('Unknown Optimizer: ' + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = Momentum(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
        return optimizer

    def _check_time(self, start=False):
        """
        记录时间用，self.time保存了[起始时间，上一步时间]
        :param start: 是否开始计时
        :return: 上一步到当前位置的时间
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def batches_add_control(self, batches, train):
        """
        向所有batch添加一些控制信息比如'dropout'
        :param batches: 所有batch的list，由DataProcessor产生
        :param train: 是否是训练阶段
        :return: 所有batch的list
        """
        for batch in batches:
            batch['train'] = train
            batch['dropout'] = self.dropout if train else self.no_dropout
        return batches

    def predict(self, model, data, data_processor):
        """
        预测，不训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :return: prediction 拼接好的 np.array
        """
        batches = data_processor.prepare_batches(data, self.eval_batch_size,
                                                 train=False)
        batches = self.batches_add_control(batches, train=False)
        model.eval()
        predictions = []
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1,
                          desc='Predict'):
            prediction = model.predict(batch)['prediction']
            predictions.append(prediction.detach())
        predictions = np.concatenate(predictions)
        sample_ids = np.concatenate([b[cfg.K_SAMPLE_ID] for b in batches])
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[cfg.
                               K_SAMPLE_ID]])
        return predictions

    def fit(self, model, data, data_processor, epoch=-1):
        """
        训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :param epoch: 第几轮
        :return: 返回最后一轮的输出，可供self.check函数检查一些中间结果
        """
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        batches = data_processor.prepare_batches(data, self.batch_size, train=True)
        batches = self.batches_add_control(batches, train=True)
        batch_size = (self.batch_size if data_processor.rank == 0 else self.batch_size * 2)
        model.train()
        accumulate_size = 0
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            accumulate_size += len(batch['Y'])
            model.optimizer.zero_grad()
            output_dict = model(batch)
            loss = output_dict['loss'] + model.l2() * self.l2_weight
            loss.backward(retain_graph=True)
            clip_grad_value_(model.parameters(), 50)
            if accumulate_size >= batch_size or batch is batches[-1]:
                model.optimizer.step()
                accumulate_size = 0
        model.eval()
        return output_dict

    def eva_termination(self, model):
        """
        检查是否终止训练，基于验证集
        :param model: 模型
        :return: 是否终止训练
        """
        metric = self.metrics[0]
        valid = self.valid_results
        if (len(valid) > 20 and metric in utils.LOWER_METRIC_LIST and utils
                .strictly_increasing(valid[-5:])):
            return True
        elif len(valid
                 ) > 20 and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(
            valid[-5:]):
            return True
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > 20:
            return True
        return False

    def train(self, model, data_processor, skip_eval=0):
        """
        训练模型
        :param model: 模型
        :param data_processor: DataProcessor实例
        :param skip_eval: number of epochs to skip for evaluations
        :return:
        """
        train_data = data_processor.get_train_data(epoch=-1)
        validation_data = data_processor.get_validation_data()
        test_data = data_processor.get_test_data()
        # saved_model=paddle.DataParallel(saved_model)
        self._check_time(start=True)

        init_train = self.evaluate(model, train_data, data_processor,
                                   metrics=['rmse', 'mae']) if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = self.evaluate(model, validation_data, data_processor
                                   ) if validation_data is not None else [-1.0] * len(self.metrics)
        init_test = self.evaluate(model, test_data, data_processor
                                  ) if test_data is not None else [-1.0] * len(self.metrics)
        logging.info('Init: \t train= %s validation= %s test= %s [%.1f s] ' %
                     (utils.format_metric(init_train), utils.format_metric(
                         init_valid), utils.format_metric(init_test), self._check_time()
                      ) + ','.join(self.metrics))
        try:
            for epoch in range(self.epoch):
                self._check_time()
                epoch_train_data = data_processor.get_train_data(epoch=epoch)
                last_batch = self.fit(model, epoch_train_data, data_processor, epoch=epoch)
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.
                        check_epoch == 0):
                    self.check(model, last_batch)
                training_time = self._check_time()
                if epoch >= skip_eval:
                    train_result = self.evaluate(model, train_data,
                                                 data_processor, metrics=['rmse', 'mae']
                                                 ) if train_data is not None else [-1.0] * len(self.
                                                                                               metrics)
                    valid_result = self.evaluate(model, validation_data,
                                                 data_processor) if validation_data is not None else [
                                                                                                         -1.0] * len(
                        self.metrics)
                    test_result = self.evaluate(model, test_data,
                                                data_processor) if test_data is not None else [-1.0
                                                                                               ] * len(self.metrics)
                    testing_time = self._check_time()
                    self.train_results.append(train_result)
                    self.valid_results.append(valid_result)
                    self.test_results.append(test_result)
                    logging.info(
                        'Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] '
                        % (epoch + 1, training_time, utils.format_metric(
                            train_result), utils.format_metric(valid_result),
                           utils.format_metric(test_result), testing_time) +
                        ','.join(self.metrics))
                    if utils.best_result(self.metrics[0], self.valid_results
                                         ) == self.valid_results[-1]:
                        model.save_model()
                    if self.eva_termination(model) and self.early_stop == 1:
                        logging.info(
                            'Early stop at %d based on validation result.' %
                            (epoch + 1))
                        break
                if epoch < skip_eval:
                    logging.info('Epoch %5d [%.1f s]' % (epoch + 1,
                                                         training_time))
        except KeyboardInterrupt:
            logging.info('Early stop manually')
            save_here = input('Save here? (1/0) (default 0):')
            if str(save_here).lower().startswith('1'):
                model.save_model()
        best_valid_score = utils.best_result(self.metrics[0], self.
                                             valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info(
            'Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] '
            % (best_epoch + 1, utils.format_metric(self.train_results[
                                                       best_epoch]), utils.format_metric(self.valid_results[best_epoch
                                                                                         ]),
               utils.format_metric(self.test_results[best_epoch]), self.
               time[1] - self.time[0]) + ','.join(self.metrics))
        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info(
            'Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] ' %
            (best_epoch + 1, utils.format_metric(self.train_results[
                                                     best_epoch]), utils.format_metric(self.valid_results[best_epoch
                                                                                       ]),
             utils.format_metric(self.test_results[best_epoch]), self.
             time[1] - self.time[0]) + ','.join(self.metrics))
        model.load_model()

    def evaluate(self, model, data, data_processor, metrics=None):
        """
        evaluate模型效果
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor
        :param metrics: list of str
        :return: list of float 每个对应一个 metric
        """
        if metrics is None:
            metrics = self.metrics
        predictions = self.predict(model, data, data_processor)
        return model.evaluate_method(predictions, data, metrics=metrics)

    def check(self, model, out_dict):
        """
        检查模型中间结果
        :param model: 模型
        :param out_dict: 某一个batch的模型输出结果
        :return:
        """
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.
                                         array2string(d, threshold=20)]) + os.linesep)
        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))
        if not loss.abs() * 0.005 < l2 < loss.abs() * 0.1:
            logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (
                loss, l2))

    def accuracy_calc(self, p, l):
        """
        calculate the accuracy with each bit flip
        :param p: predicted value
        :param l: ground truth value calculated by expression_evaluator
        :return: accuracy rate
        """
        return accuracy_score(l, p)

    def _data_reformat(self, data, bit_reverse_indices):
        """
        update the x_tag
        :param data: data dictionary
        :param bit_reverse_indices: a list with the indices of the bit to be reversed
        :return:
        """
        new_data = copy.deepcopy(data)
        for tag in new_data[cfg.C_HISTORY_POS_TAG]:
            for index in bit_reverse_indices:
                tag[index] = 1 - tag[index]
        return new_data

    def _boolean_evaluate(self, model, data, data_processor, bit_reverse_index
                          ):
        new_data = self._data_reformat(data, bit_reverse_index)
        batches = data_processor.prepare_batches(new_data, self.
                                                 eval_batch_size, train=False)
        batches = self.batches_add_control(batches, train=False)
        predictions = []
        interims = []
        model.eval()
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1,
                          desc='Predict'):
            result = model.predict(batch)
            prediction = result['prediction']
            interim = result['interim']
            interims.append(interim.detach())
            predictions.append(prediction.detach())
        predictions = np.concatenate(predictions)
        interims = np.concatenate(interims, axis=0)
        sample_ids = np.concatenate([b[cfg.K_SAMPLE_ID] for b in batches])
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[cfg.
                               K_SAMPLE_ID]])
        reorder_dict_2 = dict(zip(sample_ids, interims))
        interims = np.array([reorder_dict_2[i] for i in data[cfg.
                            K_SAMPLE_ID]])
        return predictions, interims

    @staticmethod
    def _enum_subsets(input_set):
        """
        enumerate all the subsets of given input_set
        return: a dictionary with key for the number of elements in the subsets and
        value is a list of elements
        """
        result_dict = {}
        for i in range(1, len(input_set) + 1):
            tmp_list = list(map(list, itertools.combinations(input_set, i)))
            result_dict[i] = tmp_list
        return result_dict

    @staticmethod
    def _gen_prediction_dict(p, data):
        df = pd.DataFrame()
        df['uid'] = data['uid']
        df['iid'] = data['iid']
        df['p'] = p
        df = df.sort_values(by='p', ascending=False)
        df_group = df.groupby('uid')
        y_dict = {}
        for uid, group in df_group:
            tmp_iid = group['iid'].tolist()[:1][0]
            y_dict[uid] = tmp_iid
        return y_dict

    @staticmethod
    def _accuracy_calc_from_dict(original_dict, updated_dict):
        assert len(original_dict) == len(updated_dict)
        counter = 0
        for key in original_dict:
            if updated_dict[key] == original_dict[key]:
                counter += 1
        return counter, len(original_dict)

    @staticmethod
    def _statistic_info(data):
        path = './ml100k_freq_info.pkl'
        with open(path, 'rb') as file:
            item_dict = pickle.load(file)
        tmp_list = []
        for key in data:
            tmp_list.append(item_dict[data[key]])
        tmp_list = np.array(tmp_list)
        logging.info('\n average frequency: %.1f' % tmp_list.mean())
        logging.info('\n max frequency: %.1f' % tmp_list.max())
        logging.info('\n min frequency: %.1f' % tmp_list.min())

    @staticmethod
    def _statistic_of_difference(original, updated):
        path = './ml100k_freq_info.pkl'
        with open(path, 'rb') as file:
            item_dict = pickle.load(file)
        unchanged_dict = {}
        changed_dict = {}
        for key in original:
            if original[key] == updated[key]:
                unchanged_dict[original[key]] = item_dict[original[key]]
            else:
                changed_dict[key] = {original[key]: item_dict[original[key]
                ], updated[key]: item_dict[updated[key]]}
        unchanged_freq_max = max(unchanged_dict, key=unchanged_dict.get)
        unchanged_freq_min = min(unchanged_dict, key=unchanged_dict.get)
        unchanged_freq_mean = np.array([unchanged_dict[k] for k in
                                        unchanged_dict]).mean()
        logging.info('unchanged_freq_max: {}'.format(unchanged_dict[
                                                         unchanged_freq_max]))
        logging.info('unchanged_freq_min: {}'.format(unchanged_dict[
                                                         unchanged_freq_min]))
        logging.info('unchanged_freq_mean: {}'.format(unchanged_freq_mean))
        return unchanged_dict, changed_dict

    def boolean_test(self, model, data, data_processor):
        """
        reverse bits to test the boolean sensitivity
        :param model: saved_model name
        :param data: data to use
        :param data_processor: data processor
        :return:
        """
        length_dict = {}
        lengths = [len(x) for x in data[cfg.C_HISTORY]]
        for idx, l in enumerate(lengths):
            if l not in length_dict:
                length_dict[l] = []
            length_dict[l].append(idx)
        lengths = list(length_dict.keys())
        result_dict = {}
        counter_dict = {}
        info_dict = {}
        for l in tqdm(lengths, leave=False, ncols=100, mininterval=1, desc= \
                'Prepare Batches'):
            rows = length_dict[l]
            tmp_data = {}
            for key in data:
                if data[key].dtype == np.object:
                    tmp_data[key] = np.array([np.array(data[key][r]) for r in
                                              rows])
                else:
                    tmp_data[key] = data[key][rows]
            expression_length = len(tmp_data[cfg.C_HISTORY][0])
            index_set = [i for i in range(expression_length)]
            index_sets_dict = self._enum_subsets(index_set)
            tmp_interim = None
            for key in index_sets_dict:
                acc_counter = 0
                acc_len = 0
                acc_sim = 0
                sim_counter = 0
                for index_list in index_sets_dict[key]:
                    p = self.predict(model, tmp_data, data_processor)
                    original_predict = self._gen_prediction_dict(p, tmp_data)
                    predictions, interims = self._boolean_evaluate(model,
                                                                   tmp_data, data_processor, index_list)
                    updated_predict = self._gen_prediction_dict(predictions,
                                                                tmp_data)
                    if tmp_interim is None:
                        tmp_interim = copy.deepcopy(interims)
                    else:
                        acc_sim += paddle.nn.functional.cosine_similarity(
                            paddle.to_tensor(tmp_interim), paddle.to_tensor
                            (interims), axis=-1).mean()
                        tmp_interim = copy.deepcopy(interims)
                        sim_counter += 1
                    self._statistic_info(original_predict)
                    unchanged_dict, changed_dict = (self.
                                                    _statistic_of_difference(original_predict,
                                                                             updated_predict))
                    print(asasd)
                    tmp_counter, tmp_len = self._accuracy_calc_from_dict(
                        original_predict, updated_predict)
                    acc_counter += tmp_counter
                    acc_len += tmp_len
                    tmp_str = ' '.join([str(e) for e in index_list])
                    if tmp_str not in info_dict:
                        info_dict[tmp_str] = tmp_counter / tmp_len
                accuracy = acc_counter / acc_len
                similarity = acc_sim / sim_counter
                if key not in result_dict:
                    result_dict[key] = {'accuracy': accuracy, 'similarity':
                        similarity}
                    counter_dict[key] = 1
                else:
                    result_dict[key]['accuracy'] += accuracy
                    result_dict[key]['similarity'] += similarity
                    counter_dict[key] += 1
        for key in result_dict:
            logging.info(
                '{} bit reverse average accuracy: {}\taverage similarity: {}'
                    .format(str(key), result_dict[key]['accuracy'] /
                            counter_dict[key], result_dict[key]['similarity'] /
                            counter_dict[key]))
        logging.info('----------- Details ------------')
        for key in info_dict:
            logging.info(str(key) + ': ' + str(info_dict[key]))
