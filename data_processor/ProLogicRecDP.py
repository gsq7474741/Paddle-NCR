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

import logging

import numpy as np

from configs import cfg
from data_processor.DataProcessor import DataProcessor
from data_processor.HisDataProcessor import HisDataProcessor


class ProLogicRecDP(HisDataProcessor):
    data_columns = ['X', cfg.C_HISTORY, cfg.C_HISTORY_POS_TAG,
                    cfg.C_HISTORY_LENGTH]

    def format_data_dict(self, df):
        """
        除了常规的uid,iid,label,user、item、context特征外，还需处理历史交互
        :param df: 训练、验证、测试df
        :return:
        """
        his_list = df[cfg.C_HISTORY].apply(lambda x: x.split(','))
        his_length = his_list.apply(lambda x: 0 if x[0] == '' else len(x))
        his_length = his_length[his_length > 0]
        df, his_list = df.loc[his_length.index], his_list.loc[his_length.index]
        data_dict = DataProcessor.format_data_dict(self, df)
        history_pos_tag = his_list.apply(lambda x: [(0 if i.startswith('~')
             else 1) for i in x])
        history = his_list.apply(lambda x: [(int(i[1:]) if i.startswith('~'
            ) else int(i)) for i in x])
        data_dict[cfg.C_HISTORY] = history.values
        data_dict[cfg.C_HISTORY_POS_TAG] = history_pos_tag.values
        data_dict[cfg.C_HISTORY_LENGTH] = np.array([len(h) for h in
                                                    data_dict[cfg.C_HISTORY]])
        return data_dict

    def get_boolean_test_data(self):
        logging.info('Prepare Boolean Test Data...')
        df = self.data_loader.test_df
        self.boolean_test_data = self.format_data_dict(df)
        self.boolean_test_data[cfg.K_SAMPLE_ID] = np.arange(0, len(
            self.boolean_test_data['Y']))
        return self.boolean_test_data
